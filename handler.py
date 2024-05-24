import os
import re
import abc
import sys
import json
import time
import requests
from tqdm import tqdm
from threading import Thread

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, MessagesPlaceholder

from base import log, invoke

class Handler(abc.ABC):
    # 核心信息
    qa: list = []
    error: list = []
    blocks: list = []
    # 输入输出信息
    input_path: str = ''
    output_name: str = ''
    # 配置读取信息
    model: str = None
    prompt: str = ''
    categories: list = []
    # 暂存信息
    index: int = -1
    fin_max: int = -1
    fin_dic: dict = {}
    start_time: float = 0
    max_process: int = 0
    # 线程管理
    threads: list = []
    threads_num: int = 5
    is_exit: bool = False
    # 日志管理
    log = log(__name__)

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_name = os.path.splitext(output_path)[0]
        config = json.load(open('./config/qa.json', 'r', encoding='utf-8'))
        self.prompt = self.get_prompt()
        self.categories = config['categories']
    
    def save_temp(self, qa_list:list):
        '''
        传入：新增的 QA 列表
        保存暂存信息，包括：
            output_name.jsonl 的 QA 列表
            output_name.temp 的暂存进度信息
        两个文件。会根据当前的 self.fin_max 同步修改 self.index
        '''
        # 线程读写冲突
        self.index = self.fin_max
        with open(f'{self.output_name}.jsonl', 'a', encoding='utf-8') as opt:
            for qa in qa_list:
                opt.write(json.dumps(qa, ensure_ascii=False)+'\n')
        with open(f'{self.output_name}.temp', 'w', encoding='utf-8') as opt:
            opt.write(str(self.index))
    
    @abc.abstractmethod
    def __full_json__(self) -> dict:
        '''应返回所有需要保存的对象'''
        pass

    def save_result(self):
        '''
            从缓存中读取所有 QA 对，并将所有信息写入 output_name-save.json
        '''
        if os.path.exists(f'{self.output_name}.jsonl'):
            self.qa = []
            with open(f'{self.output_name}.jsonl', 'r', encoding='utf-8') as ipt:
                for line in ipt.readlines():
                    self.qa.append(json.loads(line))
        json.dump(
            obj=self.__full_json__(),
            fp=open(f'{self.output_name}-save.json', 'w', encoding='utf-8'),
            ensure_ascii=False
        )
    
    @abc.abstractmethod
    def read(self) -> bool:
        '''读取文件，若文件不完整，则返回 False，若读取成功，则返回 True'''
        pass
    @abc.abstractmethod
    def initialize(self):
        '''初始化文件'''
        pass

    @staticmethod
    def message_type(s:str):
        if s == 'HumanMessage':
            return 'user'
        if s == 'SystemMessage':
            return 'system'
        if s == 'AIMessage':
            return 'assistant'
        else:
            return 'unknown'
    @abc.abstractmethod
    def get_prompt(self):
        '''返回生成 QA 对所需的提示词'''
        pass
    @abc.abstractmethod
    def run(self, block:list, idx:int, retry=3):
        '''具体的模型调用和校验，负责调用 self.finish 方法'''
        pass
    
    def finish(self, idx:int, qa_list:list):
        '''负责进度记录和QA的转存'''
        self.fin_dic[idx] = True
        while self.fin_max+1 in self.fin_dic:
            self.fin_max += 1
        self.save_temp(qa_list)
    
    def show_process(self):
        '''显示当前进度'''
        if self.is_exit:
            return
        
        s = int(time.time()-self.start_time)
        s = f'{s//3600}:{(s%3600)//60:02d}:{s%60:02d}'
        threads = ', '.join([str(t[1]+1) for t in self.threads])
        percent = int((self.fin_max+1) / len(self.blocks) * 100)
        s = f'\r[{self.fin_max+1}/{len(self.blocks)}]({percent}%) | threads: [{threads}] | {s}'
        self.max_process = max(self.max_process, len(s))
        sys.stdout.write('\r'+' '*self.max_process+'\r')
        sys.stdout.flush()
        sys.stdout.write(s)
        sys.stdout.flush()
    
    def generate(self):
        '''生成 QA 对'''
        self.fin_max, self.fin_dic = self.index, {}

        self.threads = []
        self.start_time = time.time()
        for i in range(self.index+1, len(self.blocks), 1):
            if self.is_exit:
                break

            new_thread = Thread(
                target=self.run,
                args=(self.blocks[i], i, )
            )
            new_thread.start()
            self.threads.append((new_thread, i))

            while len(self.threads) >= self.threads_num:
                time.sleep(1)
                temp = []
                for t in self.threads:
                    if t[0].is_alive():
                        temp.append(t)
                    else:
                        del t
                self.threads = temp
                self.show_process()
        print()
        for t in self.threads:
            t[0].join()
        self.save_result()
    
    def exit(self):
        '''退出程序，并保存当前处理状态'''
        self.is_exit = True
        for t in self.threads:
            t[0].join()
        self.save_result()
        
    def handle(self, init:bool=None, model:str=None):
        '''调用此函数一键处理消息文件'''
        self.model = model
        os.makedirs(os.path.dirname(self.output_name), exist_ok=True)
        if os.path.exists(f'{self.output_name}.temp'):
            with open(f'{self.output_name}.temp', 'r', encoding='utf-8') as ipt:
                self.index = int(ipt.readline())
        if init or not self.read():
            self.initialize()

        self.log.log(f'初始化完成，正在启动生成QA对，开始时进度：{self.index+1}/{len(self.blocks)}')
        self.generate()
        if self.is_exit:
            self.log.log('程序退出成功！')
        else:
            self.log.log(f'QA对生成完成，生成QA对：{len(self.qa)} 个')
            self.log.log(f'{self.input_path} 处理完成，输出至：{self.output_name}-save.json')
    
    def __status__(self) -> str:
        '''返回当前处理状态'''
        return f'处理文件：{self.input_path}\n当前处理进度：{self.index+1} / {len(self.blocks)}'