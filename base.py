import time
import json
import random
import requests

# 阿里灵积 SDK
import dashscope
from http import HTTPStatus

# 百度千帆 SDK
import qianfan

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, MessagesPlaceholder

api_keys = json.load(open('../../api_key.json', 'r', encoding='utf-8'))

class log:
    name: str

    def __init__(self, name:str):
        self.name = name
        
    def log(self, msg, level='I'):
        '''记录信息'''
        level = {'I': 'INFO', 'E': 'ERROR', 'W': 'WARNING'}.get(level, 'INFO')
        msg = f'[{level}] [{time.strftime(r"%Y-%m-%d %H:%M:%S", time.localtime())}] {msg}'
        with open(f'./log/{self.name}.log', 'a', encoding='utf-8') as f:
            f.write(msg+'\n')
        print(f'<{self.name}> {msg}')

local_llm, moonshot_llm = None, None
def local_invoke(prompt, data:dict) -> str:
    '''调用本地模型'''
    global local_llm
    if local_llm is None:
        local_llm = ChatOpenAI(
            model_name='Qwen',
            openai_api_base='http://localhost:8001/v1',
            openai_api_key='EMPTY',
            streaming=False
        )
    return (prompt | local_llm).invoke(data).content

def moonshot_invoke(prompt, data:dict, model:str) -> str:
    '''调用 KimiChat 模型'''
    global moonshot_llm, api_keys
    if moonshot_llm is None:
        moonshot_llm = ChatOpenAI(
            model_name=model,
            base_url='https://api.moonshot.cn/v1',
            api_key=api_keys['kimi'],
        )
    return (prompt | moonshot_llm).invoke(data).content

def dashscope_invoke(prompt, data:dict, model:str) -> str:
    '''调用阿里灵积平台模型'''
    global api_keys
    def message_type(s:str):
        '''将消息类型转换为角色名'''
        if s == 'HumanMessage':
            return 'user'
        if s == 'SystemMessage':
            return 'system'
        if s == 'AIMessage':
            return 'assistant'
        else:
            return 'unknown'

    message = prompt.invoke(data).to_messages()
    messages = [{
        'role': message_type(type(i).__name__),
        'content': i.content.strip()
    } for i in message]
    
    # print('调用 token 数：{}'.format(len(prompt.invoke(data).to_string())))
    # return ''

    dashscope.api_key = api_keys['dashscope']
    response = dashscope.Generation.call(
        model = model,
        messages = messages,
        seed = random.randint(1, 10000),
        result_format = 'text'
    )
    if response.status_code == HTTPStatus.OK:
        return response.output.text
    else:
        return ''

def minimax_invoke(prompt, data:dict, model:str) -> str:
    '''调用 Minimax 模型'''
    global api_keys
    def message_type(s:str):
        '''将消息类型转换为角色名'''
        if s == 'HumanMessage':
            return 'user'
        if s == 'SystemMessage':
            return 'system'
        if s == 'AIMessage':
            return 'assistant'
        else:
            return 'unknown'
    
    url = 'https://api.minimax.chat/v1/text/chatcompletion_v2'
    headers = {
        'Authorization': f'Bearer {api_keys["minimax"]}',
        'Content-Type': 'application/json'
    }
    message = prompt.invoke(data).to_messages()
    messages = [{
        'role': message_type(type(i).__name__),
        'content': i.content.strip()
    } for i in message]
    body = {
        'model': model,
        'stream': False,
        # 'temperature': random.random(),
        'messages': messages,
        'tool_choice': 'none'
    }
    response = requests.post(url=url, headers=headers, json=body)
    try:
        res = response.json()
        if res['base_resp']['status_code'] == 0 \
            and res['choices'][0]['finish_reason'] == 'stop':
            return res['choices'][0]['message']['content']
        else:
            raise ValueError(f'返回值错误，详细返回：{res}')
    except Exception as e:
        # print(response.text, e)
        return ''

def qianfan_invoke(prompt, data:dict, model:str) -> str:
    '''调用百度千帆平台模型'''
    global api_keys
    def message_type(s:str):
        '''将消息类型转换为角色名'''
        if s == 'HumanMessage':
            return 'user'
        if s == 'SystemMessage':
            return 'system'
        if s == 'AIMessage':
            return 'assistant'
        else:
            return 'unknown'
    
    message = prompt.invoke(data).to_messages()
    system_prompt = ''
    if message_type(type(message[0]).__name__) == 'system':
        system_prompt = message[0].content.strip()
        message = message[1:]
    messages = [{
        'role': message_type(type(i).__name__),
        'content': i.content.strip()
    } for i in message]

    chat_completion = qianfan.ChatCompletion(
        ak=api_keys['qianfan']['ak'],
        sk=api_keys['qianfan']['sk']
    )
    response = chat_completion.do(
        model=model,
        messages=messages,
        system=system_prompt
    )
    return response['body']['result']

default_online = 'qwen-plus'
def invoke(prompt, data:dict, online:str=None) -> str:
    '''
        调用模型

        参数：
            prompt: 调用模型的提示
            data: 调用模型的数据
            online: 在线模型名称，
                如果为不填则使用默认的在线模型，
                为空字符串则使用本地模型
        返回：
            字符串，模型输出
    '''
    if online is None:
        online = default_online

    if online != '':
        # 使用阿里灵积平台的模型
        if online in ['qwen1.5-14b-chat', 'qwen-turbo', 'qwen-plus', 'qwen-max']:
            return dashscope_invoke(prompt, data, online)
        # 使用 Minimax 模型
        elif online in ['abab6-chat', 'abab5.5-chat', 'abab5.5s-chat']:
            return minimax_invoke(prompt, data, online)
        # 使用 KimiChat 模型
        elif online in ['moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k']:
            return moonshot_invoke(prompt, data, online)
        # 使用百度千帆平台的模型
        elif online in ['ERNIE-Bot-4', 'ERNIE-Bot']:
            return qianfan_invoke(prompt, data)
        else:
            return ''
    else:
        return local_invoke(prompt, data)

if __name__ == '__main__':
    print(invoke(ChatPromptTemplate.from_messages([
        HumanMessagePromptTemplate.from_template('''
你是一个提示工程师，现在你需要拟写一段提示词，完成以下任务：模型将收到若干个QA对，模型需要为QA对的质量进行评估，具有如下标准的QA对应该被称为“劣”：
-   回答提供的信息不足以解决问题。
-   回答中存在指代，使得单独的回答并不能很好的提供解决问题的方法。（比如：“你可以通过 @ 某某人解决问题”）
-   模型因为提供信息不足，结合自己的知识做出的过于宽泛的回答。（比如：“在大学中，遇到这种问题一般可以咨询对应自己学校的教务处”）
- 问回答中出现个人信息，不具备可参考性的问题。（比如：“翠翠是学长还是学姐？”）

模型的输出应该为一个列表，内容为输入的QA对个数个“优”或“劣”。
下面，请你开始拟写提示词！
''')
    ]), {}, 'moonshot-v1-8k'))