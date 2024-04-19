import time
import random
import dashscope
from http import HTTPStatus
from langchain.chat_models import ChatOpenAI

class log:
    name: str

    def __init__(self, name):
        self.name = name
        
    def info(self, msg):
        # print('[{}] {}'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))
        with open(f'./log/{self.name}.log', 'a', encoding='utf-8') as f:
            f.write('[INFO][{}] {}\n'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), msg))

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
            model = 'qwen-plus',
            # model = 'qwen-max',
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