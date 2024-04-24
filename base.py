import time
import random
import dashscope
from http import HTTPStatus
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, MessagesPlaceholder

class log:
    name: str

    def __init__(self, name):
        self.name = name
        
    def info(self, msg):
        # print('[{}] {}'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))
        with open(f'./log/{self.name}.log', 'a', encoding='utf-8') as f:
            f.write('[INFO][{}] {}\n'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))
        
    def warning(self, msg):
        # print('[{}] {}'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))
        with open(f'./log/{self.name}.log', 'a', encoding='utf-8') as f:
            f.write('[WARNING][{}] {}\n'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))

llm = ChatOpenAI(
    model_name='Qwen',
    openai_api_base='http://localhost:8001/v1',
    openai_api_key='EMPTY',
    streaming=False
)

default_online = True
def invoke(prompt, data:dict, online:bool=None) -> str:
    '''调用模型'''

    def message_type(s:str):
        if s == 'HumanMessage':
            return 'user'
        if s == 'SystemMessage':
            return 'system'
        if s == 'AIMessage':
            return 'assistant'
        else:
            return 'unknown'

    if online is None:
        online = default_online

    if online:
        message = prompt.invoke(data).to_messages()
        messages = [{
            'role': message_type(type(i).__name__),
            'content': i.content.strip()
        } for i in message]

        # print('调用 token 数：{}'.format(len(prompt.invoke(data).to_string())))
        # return ''
        
        response = dashscope.Generation.call(
            # model = 'qwen1.5-14b-chat',
            # model = 'qwen-turbo',
            # model = 'qwen-plus',
            model = 'qwen-max',
            messages = messages,
            seed = random.randint(1, 10000),
            result_format = 'text'
        )
        if response.status_code == HTTPStatus.OK:
            return response.output.text
        else:
            return ''
    else:
        return (prompt | llm).invoke(data).content

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
    ]), {}))