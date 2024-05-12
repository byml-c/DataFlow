import os
import re
import json
import time
from tqdm import tqdm
from typing import Literal
from threading import Thread

import traceback

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain.document_loaders import TextLoader, BSHTMLLoader, PyPDFLoader, Docx2txtLoader, CSVLoader

from base import log, invoke
'''
    需要安装的包：
    pip install pypdf docx2txt bs4
'''

class ContentBlock:
    size: int = 0
    content: list[(str, int, bool)] = []
    type: Literal['ContentBlock'] = 'ContentBlock'

    def __init__(self, content:str='', integrity:bool=False):
        '''
            content: 内容
            integrity: 是否允许拆分
        '''
        self.size = 0
        self.content = []
        if content != '':
            self.append_content(content, len(content), integrity)
    
    def __str__(self) -> str:
        return f'[size: {self.size}] '+self.content.__str__()

    def to_string(self) -> str:
        return ''.join([i[0] for i in self.content])

    def append_content(
        self,
        content: str,
        size: int,
        integrity: bool=False
    ) -> None:
        self.content.append((content, size, integrity))
        self.size += size
    
    def merge_block(
        self,
        item: Literal['ContentBlock']
    ):
        '''
            合并两个 ContentBlock 对象，将 item 的内容加入到 self 后面
        '''
        self.content += item.content
        self.size += item.size

    def append_block(
        self,
        item: Literal['ContentBlock'],
        block_size: int
    ) -> Literal['ContentBlock']:
        '''
            在内容后追加字符串，限制最大长度为 block_size

            返回：剩余的文本构成的 ContentBlock
            注意：
                如果 item 被完全加入，rest 为 None
                如果恰好分割到的文本块的 integrity 为 True，且长度不够完整加入，将会选择不加入这个文本块
        '''
        if self.size + item.size <= block_size:
            self.content += item.content
            self.size += item.size
            return None
        else:
            accept, rest, full = ContentBlock(), ContentBlock(), False
            for it in item.content:
                if full:
                    # 此块已满，不加入
                    rest.append_content(it[0], it[1], it[2])
                else:
                    if self.size + it[1] <= block_size:
                        accept.append_content(it[0], it[1], it[2])
                    else:
                        full = True
                        if it[2]:
                            # 此块不可拆分，故不加入
                            rest.append_content(it[0], it[1], it[2])
                        else:
                            # 此块可拆分，拆分一部分加入
                            split_size = block_size - self.size
                            accept.append_content(it[0][0 : split_size], split_size, False)
                            rest.append_content(it[0][split_size : ], it[1] - split_size, False)
            self.merge_block(accept)
            return rest

    def append_keep_block(
        self,
        item: Literal['ContentBlock'],
        block_size: int
    ):
        '''
            通过删除靠前的内容，保持文本总长度不大于 block_size 大小
        '''
        self.merge_block(item)
        if self.size <= block_size:
            return 
        else:
            while self.size - self.content[0][1] > block_size:
                self.size -= self.content[0][1]
                self.content.pop(0)
            if self.content[0][2]:
                # 不可拆分的文本块，直接整块删除
                self.size -= self.content[0][1]
                self.content.pop(0)
            else:
                # 可拆分的文本块，删除前一部分，保持总长度不大于 block_size
                remove_size = self.size - block_size
                self.size = block_size
                self.content[0] = (self.content[0][0][remove_size : ], self.content[0][1] - remove_size, False)

class QAJsonLoader(BaseLoader):
    '''
        转换用于 Qwen 微调的 json 格式 QA 对数据
    '''

    docs = []
    Q, A = '', ''
    def __init__(self, file_path, encoding):
        self.file_path = file_path
        self.encoding = encoding
    
    def load(self) -> list[Document]:
        try:
            with open(self.file_path, 'r', encoding=self.encoding) as file_input:
                content = file_input.read()
            return Document(page_content=content, metadata={'source': self.file_path})
        except Exception as e:
            raise RuntimeError(f'Error loading {self.file_path}') from e
    
    def load_and_split(self, block_size, cover_size, text_splitter):
        self.docs = []
        qa_list = json.load(
            open(self.file_path, 'r', encoding=self.encoding))
        
        for page in range(len(qa_list)):
            qa = qa_list[page]
            self.Q = text_splitter.filter(qa['conversations'][0]['value'])
            page_size = block_size - len(self.Q) - 10
            text_splitter.block_size = page_size
            text_splitter.cover_size = cover_size
            self.A = text_splitter.filter(qa['conversations'][1]['value'])
            self.A = text_splitter.content_splitter(
                doc = self.A, metadata = {'source': self.file_path, 'page': page+1}
            )
            for doc in self.A:
                doc.page_content = self.Q + doc.page_content
                self.docs.append(doc)

        return self.docs

class TextSplitter:
    index = False
    block_size = 512
    cover_size = 128

    def __init__(self, block_size, cover_size, index=False):
        self.index = index
        self.block_size = block_size
        self.cover_size = cover_size
    
    def invisible_filter(self, content:str) -> str:
        content = re.sub(r'[\t\f\v\r \xa0]+', ' ', content)
        content = re.sub(r'[\n]+', '\n', content)
        return content
    
    def tag_filter(self, content:str) -> str:
        return content

    def filter(self, content:str) -> str:
        '''
            对内容进行过滤，去除不可见字符、HTML标签等
        '''
        content = content.strip()
        content = self.invisible_filter(content)
        content = self.tag_filter(content)
        return content
    
    def url_splitter(self, doc:str) -> list[ContentBlock]:
        '''
            以链接划分字符串
        '''
        url_pattern = r'(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)'
        doc = re.sub(url_pattern, r'$url$\1$url$', doc)
        docs = [i.strip() for i in doc.split('$url$')]
        docs = [ContentBlock(i, i[0:4] == 'http') for i in docs]
        return docs

    def content_splitter(self, doc:str, metadata:dict) -> list[Document]:
        '''
            对内容划分，遵循 block_size 和 cover_size
            期望最大程度上保持链接完整

            返回：划分后的 Document 列表
        '''
        # return [Document(page_content=doc, metadata=metadata)]
    
        docs, index_count = [], 0
        blocks = self.url_splitter(doc)
        last, this = ContentBlock(), ContentBlock()
        for block in blocks:
            result = this.append_block(block, self.block_size-self.cover_size)
            while result is not None:
                if self.index:
                    index_count += 1
                    metadata.update({'index': index_count})
                docs.append(Document(
                    page_content = last.to_string()+this.to_string(),
                    metadata = dict(metadata)
                ))
                last.append_keep_block(this, self.cover_size)
                this = ContentBlock()
                result = this.append_block(result, self.block_size-self.cover_size)

                if result is not None and len(result.content) == 1 and result.content[0][2] \
                    and result.size > self.block_size - self.cover_size:
                    # 如果剩余的文本块不可拆分，直接加入
                    docs.append(Document(
                        page_content = result.to_string(),
                        metadata = dict(metadata)
                    ))
                    result = None
                    last = ContentBlock()
                
        if this is not None and this.size > 0:
            if self.index:
                index_count += 1
                metadata.update({'index': index_count})
            docs.append(Document(
                page_content = last.to_string()+this.to_string(),
                metadata = dict(metadata)
            ))
        return docs

    def split_documents(self, docs:list[Document]) -> list[Document]:
        '''
            对文档列表进行划分
        '''
        
        temp = []
        for doc in docs:
            temp += self.content_splitter(
                doc = self.filter(doc.page_content),
                metadata = doc.metadata
            )
        return temp

class DocumentHandler:
    '''
        对文档进行处理
    '''
    qa: list
    error: list
    blocks: list
    online: bool
    categories: list
    log = log(__name__)
    supported_types = ['txt', 'html', 'htm', 'md', 'pdf']

    def __init__(self):
        self.qa = []
        self.error = []
        self.blocks = []
        self.online = True

        config = json.load(open('./config/qa.json', 'r', encoding='utf-8'))
        self.categories = config['categories']
    
    def __dict__(self):
        return {
            'qa': self.qa,
            'error': self.error,
            'blocks': self.blocks
        }

    def save(self, output_path:str):
        '''保存文档数据'''
        os.makedirs(os.path.dirname(output_path), exist_ok=True) 
        json_path = os.path.splitext(output_path)[0]+'-save.json'
        json.dump(
            obj=self.__dict__(),
            fp=open(json_path, 'w', encoding='utf-8'),
            ensure_ascii=False
        )
    
    def generate_QA_by_summary_blocks(self, model:str=None):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
'''
- Role: 文档内容分析专家
- Background: 您将接收一个包含多种信息的文档，并从中提取关键信息，然后生成相关的问答对。
- Profile: 您是一位专业的文档内容分析专家，具备深厚的文本分析能力和信息提取技巧。
- Skills: 您需要运用文本解析、信息提取、问题生成和答案撰写等技能。
- Goals: 您需要从文档中识别关键信息，并围绕每个关键信息点生成相关的问答对。
- Constrains: 您生成的问题应具有针对性，能够覆盖文档内容的主要方面，答案应详细且准确，格式需符合指定的JSON结构。
- OutputFormat: 您需要按照以下JSON格式输出结果，确保每个问答对都包含“问题”、“回答”、“分类”、“关键词”和“依据”。
- Workflow:
  1. 仔细阅读提供的文档内容。
  2. 识别文档中的关键信息点。
  3. 为每个关键信息点生成相关的问答对。
  4. 确保问题和答案与文档内容相关，风格自然。
  5. 使用指定的格式输出问答对。

- Examples: 
    - 文档：
        "新开的咖啡馆提供多种饮品，其中拿铁是他们的特色推荐，以其浓郁的咖啡香和细腻的奶泡受到顾客的喜爱。"
    - QA: 
        {{
            "问题": "新开的咖啡馆的特色饮品是什么？",
            "回答": "新开的咖啡馆的特色饮品是拿铁，它以其浓郁的咖啡香和细腻的奶泡受到顾客的喜爱。",
            "分类": "校园生活服务",
            "关键词": ["咖啡馆", "特色饮品", "拿铁"],
            "依据": ["新开的咖啡馆提供多种饮品，其中拿铁是他们的特色推荐，以其浓郁的咖啡香和细腻的奶泡受到顾客的喜爱。"]
        }}

以下是您需要遵循的输出格式：
[
    {{
        "问题": "xxx",
        "回答": "xxx",
        "分类": "xxx",
        "关键词": [
            "xxx", 
            "xxx"
        ],
        "依据": [
            "xxx", 
            "xxx"
        ]
    }},
    ...
    {{
        "问题": "xxx",
        "回答": "xxx",
        "分类": "xxx",
        "关键词": [
            "xxx", 
            "xxx"
        ],
        "依据": [
            "xxx", 
            "xxx"
        ]
    }}
]

- OutputRequirement:
    - 问题、回答：问答对文本
    - 分类：所属分类，应为这些分类中的一个：{categories}
    - 关键词：根据提问生成关键词，用于检索

请根据我提供的文档内容，生成相关的QA内容。如果提供的信息不足以生成QA，请仅输出“无答案”三个字。
'''
            ), HumanMessagePromptTemplate.from_template('''文档：\n{document}''')
        ])
        
        def run(block:dict):
            while True:
                response = invoke(prompt, {
                    'document': block['page_content'],
                    'categories': '、'.join(self.categories)
                }, online=model)

                if response == '':
                    self.log.log('模型返回为空，重试！数据：\n{}'.format(block))
                    time.sleep(0.5)
                    continue
                if re.search('无答案', response):
                    self.log.log(f'无答案，加入错误列表！数据：\n{block}\n返回：\n{response}', 'E')
                    self.error.append(block)
                    break

                try:
                    res = re.search(r'```json((.|[\n\r])*)```', response, re.S)
                    res = json.loads(res.group(1) if res else response)
                    if type(res) == dict:
                        res = [res]
                    
                    for r in res:
                        for field in ['问题', '回答', '分类', '关键词', '依据']:
                            if field not in r:
                                raise ValueError('模型返回格式错误')
                        if r['分类'] not in self.categories:
                            raise ValueError('分类错误')
                        if type(r['关键词']) != list:
                            raise ValueError('模型返回格式错误')
                        if type(r['依据']) != list:
                            r['依据'] = [r['依据']]
                        format_r = {
                            'Q': r['问题'],
                            'A': r['回答'],
                            'type': r['分类'].replace(' ', ''),
                            'keywords': [i.replace(' ', '') for i in r['关键词']],
                            'refs': r['依据'],
                            'author': 'AI'
                        }
                        self.qa.append(format_r)
                    break
                except Exception as err:
                    self.log.log('出错：{}，模型返回：{}'.format(err, response))
                    time.sleep(0.5)
        
        if len(self.blocks) == 0:
            self.log.log('文档为空，无法生成QA对')
        else:
            threads = []
            thread_num = 5
            
            for i in tqdm(range(len(self.blocks))):
                new_thread = Thread(
                    target=run,
                    args=(self.blocks[i],)
                )
                new_thread.start()
                threads.append(new_thread)

                while len(threads) >= thread_num:
                    time.sleep(0.5)
                    temp = []
                    for t in threads:
                        if t.is_alive():
                            temp.append(t)
                    threads = temp
            
            for t in threads:
                t.join()

    def load(self, path, size, cover, type, encoding, index):
        '''加载文档，返回文档列表'''
        self.log.log('初始化文档数据，路径：{}'.format(path))
        self.blocks = []
        if type == 'txt':
            self.blocks = TextLoader(file_path=path, encoding=encoding). \
                load_and_split(text_splitter=TextSplitter(size, cover, index))
        elif type == 'html' or type == 'htm':
            self.blocks = BSHTMLLoader(file_path=path, open_encoding=encoding). \
                load_and_split(text_splitter=TextSplitter(size, cover, index))
        elif type == 'md':
            self.blocks = TextLoader(file_path=path, encoding=encoding). \
                load_and_split(text_splitter=TextSplitter(size, cover, index))
        elif type == 'pdf':
            self.blocks = PyPDFLoader(file_path=path). \
                load_and_split(text_splitter=TextSplitter(size, cover, index))
        self.blocks = [{
            'page_content': i.page_content,
            'metadata': i.metadata
        } for i in self.blocks]
        self.log.log('文档加载完成，共 {} 个文档'.format(len(self.blocks)))
    
    def read(self, path:str, encoding:str):
        '''从文件加载文档数据'''
        path = os.path.splitext(path)[0]+'-save.json'
        self_obj = json.load(open(path, 'r', encoding='utf-8'))

        self.qa = self_obj['qa']
        self.error = self_obj['error']
        self.blocks = self_obj['blocks']

    def handle(self, input_path, output_path='./output.txt',
               size=512, cover=64, type='auto', encoding='utf-8', index:bool=True, init:bool=None, model:str=None):
        '''调用此函数一键处理文档'''
        if init is None:
            init = not os.path.exists(output_path+'.blocks')
        
        if init:
            if type == 'auto':
                suffix = os.path.splitext(input_path)[1].lower().replace('.', '')
                type = suffix if suffix in self.supported_types else 'txt'
            self.load(input_path, size, cover, type, encoding, index)
            self.save(output_path)
        else:
            self.read(output_path, encoding)
        
        # with open(output_path+'.txt', 'w', encoding=encoding) as f_out:
        #     for doc in self.blocks:
        #         f_out.write(doc['page_content']+'\n')

        if init or len(self.qa) == 0:
            self.log.log('正在生成QA对')
            self.generate_QA_by_summary_blocks(model)
            self.save(output_path)
            self.log.log('QA对生成完成，生成QA对：{}个'.format(len(self.qa)))
        self.log.log('{} 处理完成，输出至：{}'.format(input_path, output_path))

def handle_folder(input_root, output_root):
    '''递归处理文件夹内所有文件'''
    supported_types = ['txt', 'pdf', 'md', 'html', 'htm']
    for root, dirs, files in os.walk(input_root):
        for file in files:
            suffix = os.path.splitext(file)[1].lower().replace('.', '')
            if suffix in supported_types:
                input_path = os.path.join(root, file)
                output_path = os.path.join(root, file).replace(input_root, output_root)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                try:
                    handler = DocumentHandler()
                    handler.handle(
                        input_path=input_path,
                        output_path=output_path,
                        size=1024,
                        cover=128,
                        init=True
                    )
                except Exception as e:
                    print('处理文件出错：{}，原因：{}'.format(input_path, traceback.format_exc()))
            else:
                print('文件类型不支持：{}'.format(input_path))

if __name__ == '__main__':
    data_root = '../RawData/'
    output_root = '../T_QA/'

    handle_list = [
        'Documents\南哪助手新生问答指南\教务服务\关于“推免”_“保研”.txt'
    ]

    for file_name in handle_list:
        h = DocumentHandler()
        h.handle(data_root+file_name, output_root+file_name)
    # handle_folder('./temp_TXT', './temp_TXT_output')