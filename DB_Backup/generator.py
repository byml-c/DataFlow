import os
import json
import time
from tqdm import tqdm
from message import MessageHandler
from document import DocumentHandler
from database import Database
from base import default_online

class Generator:
    uid: str
    root: str
    status: int
    db: Database
    files: list[str]

    time_format = r'%Y-%m-%d %H:%M:%S'
    valid_suffix = ['qa', 'txt', 'pdf', 'md', 'html', 'htm']

    def __init__(self, uid:str, root_path:str='../RawData', force:bool=False):
        '''初始化，如果 uid 存在则加载状态，否则创建新状态'''
        if os.path.exists(f'./log/{uid}.json') and not force:
            data = json.load(open(f'./log/{uid}.json', 'r', encoding='utf-8'))
            self.uid = uid
            self.root = root_path
            self.files = data['files']
            self.status = data['status']

            if self.status == -1:
                self.log(f'预加载状态成功，当前进度：未开始运行。')
            elif self.status < len(self.files):
                self.log(f'预加载状态成功，当前进度：{self.status+1}/{len(self.files)}')
            else:
                self.log(f'预加载状态成功，当前进度：运行已完成。')
        else:
            self.status = -1
            self.uid = uid
            self.root = root_path
            self.load_files()
            self.save()
            self.log(f'初始化成功！')
        self.db = Database(uid, 'GC_QA')
    
    def __dict__(self) -> dict:
        return {
            'uid': self.uid,
            'files': self.files,
            'status': self.status
        }

    def log(self, msg:str, level='I') -> None:
        '''记录信息'''
        level = {'I': 'INFO', 'E': 'ERROR', 'W': 'WARNING'}.get(level, 'INFO')
        msg = f'[INFO] [{time.strftime(self.time_format, time.localtime())}] {msg}'
        with open(f'./log/{self.uid}.log', 'a', encoding='utf-8') as f:
            f.write(f'{msg}\n')
        print(msg)

    def save(self) -> None:
        '''保存运行状态'''
        json.dump(
            obj=self.__dict__(),
            fp=open(f'./log/{self.uid}.json', 'w', encoding='utf-8'),
            ensure_ascii=False
        )
    
    def load_files(self) -> None:
        '''获取文件列表'''
        self.files = []
        for root, dirnames, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.split('.')[-1] in self.valid_suffix:
                    self.files.append(os.path.join(root, filename))
        self.log(f'加载文件成功！共加载 {len(self.files)} 个文件。')

    def run(self, model:str=None, retry:bool=False):
        '''启动生成器，开始生成 QA 对'''
        global default_online
        if model is None:
            model = default_online
        self.log(f'生成器启动，使用模型：{model}')

        for file_id in tqdm(range(len(self.files)), desc='<generator>'):
            if file_id <= self.status:
                continue

            file = self.files[file_id]
            output_path = f'./temp/{self.uid}/{file.replace(self.root, "")}'
            input_path = os.path.abspath(file)
            output_path = os.path.abspath(output_path)
            if file.split('.')[-1] == 'qa':
                # print(f'Call message handler input_path={file} output_path={output_path}')
                message_handler = MessageHandler()
                message_handler.handle(input_path=input_path, output_path=output_path, model=model)
            else:
                # print(f'Call document handler input_path={file} output_path={output_path}')
                document_handler = DocumentHandler()
                document_handler.handle(input_path=input_path, output_path=output_path, model=model)

            output_data = json.load(open(f'{os.path.splitext(output_path)[0]}-save.json', 'r', encoding='utf-8'))
            for data in output_data['qa']:
                data.update({
                    'source': f'./{file.replace(self.root, "")}'
                })
                self.db.insert(data)
            self.status = file_id
            self.save()

if __name__ == '__main__':
    a = Generator('BATCH01', root_path='../RunData')
    a.run('qwen1.5-32b-chat')