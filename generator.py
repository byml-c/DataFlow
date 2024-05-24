import os
import json
import time
from tqdm import tqdm
from message import MessageHandler
from document import DocumentHandler
from database import Database
from base import default_online
from threading import Thread
from base import log

class Generator:
    uid: str
    root: str
    status: int
    db: Database
    files: list[str]
    handler: None
    is_exit: bool
    logger: log

    time_format = r'%Y-%m-%d %H:%M:%S'
    valid_suffix = ['qa', 'txt', 'pdf', 'md', 'html', 'htm']

    def __init__(self, uid:str, root_path:str='../RawData', force:bool=False):
        '''初始化，如果 uid 存在则加载状态，否则创建新状态'''
        self.is_exit = False
        self.logger = log(uid)
        if os.path.exists(f'./log/{uid}.json') and not force:
            data = json.load(open(f'./log/{uid}.json', 'r', encoding='utf-8'))
            self.uid = uid
            self.root = root_path
            self.files = data['files']
            self.status = data['status']

            if self.status == -1:
                self.logger.log(f'预加载状态成功，当前进度：未开始运行。')
            elif self.status < len(self.files):
                self.logger.log(f'预加载状态成功，当前进度：{self.status+1}/{len(self.files)}')
            else:
                self.logger.log(f'预加载状态成功，当前进度：运行已完成。')
        else:
            self.status = -1
            self.uid = uid
            self.root = root_path
            self.load_files()
            self.save()
            self.logger.log(f'初始化成功！')
        self.db = Database(uid, 'GC_QA')
    
    def __dict__(self) -> dict:
        return {
            'uid': self.uid,
            'files': self.files,
            'status': self.status
        }

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
    
    def __status__(self) -> str:
        '''查看当前状态'''
        s = f'当前进度：{self.status+1}/{len(self.files)}\n'
        s += f'处理器状态：\n{self.handler.__status__()}\n'
        s += f'生成数据总量：{self.db.count()} 条\n'
        s += '-'*20+'\n'
        return s

    def exit(self):
        '''退出程序'''
        self.is_exit = True
        self.handler.exit()
        self.save()
        self.logger.log('程序退出成功！')

    def run(self, model:str=None, retry:bool=False):
        '''启动生成器，开始生成 QA 对'''
        global default_online
        if model is None:
            model = default_online
        self.logger.log(f'生成器启动，使用模型：{model}')

        length = len(self.files)
        for file_id in range(length):
            if file_id <= self.status:
                continue
            if self.is_exit:
                break

            file = self.files[file_id]
            output_path = f'./temp/{self.uid}/{file.replace(self.root, "")}'
            input_path = os.path.abspath(file).replace('\\', '/')
            output_path = os.path.abspath(output_path).replace('\\', '/')

            try:
                if file.split('.')[-1] == 'qa':
                    # print(f'Call message handler input_path={file} output_path={output_path}')
                    self.handler = MessageHandler(input_path=input_path, output_path=output_path)
                    self.handler.handle(model=model)
                else:
                    # print(f'Call document handler input_path={file} output_path={output_path}')
                    self.handler = DocumentHandler(input_path=input_path, output_path=output_path)
                    self.handler.handle(model=model)
                if self.is_exit:
                    break
                
                output_data = json.load(open(f'{os.path.splitext(output_path)[0]}-save.json', 'r', encoding='utf-8'))
                for data in output_data['qa']:
                    data.update({
                        'source': f'./{file.replace(self.root, "")}'
                    })
                    self.db.insert(data)
            except Exception as e:
                self.logger.log(f'处理文件 {file} 时出现错误：{e}', 'E')
            finally:
                if not self.is_exit:
                    self.status = file_id
                    self.save()
                    
                    if os.name == 'nt':
                        os.system('cls')
                    else:
                        os.system('clear')
                    print(f'[{int((file_id+1)/length*100)}%|{(file_id+1)}/{length}] {input_path} 处理完成！')

if __name__ == '__main__':
    generator = Generator('BATCH01', root_path='../RunData')
    main_thread = Thread(target=generator.run, args=('qwen1.5-32b-chat',))
    main_thread.start()
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

    print('加载完成，程序开始运行')
    while True:
        try:
            print('可输入 status 查看运行状态，输入 exit 终止程序')
            ipt = input('>> ')
            if ipt == 'status':
                print(generator.__status__())
            if ipt == 'exit':
                print('正在终止程序，请耐心等待 20s 左右……')
                generator.exit()
                main_thread.join()
                print('程序终止成功！')
                break
            else:
                print('无效指令！')
        except KeyboardInterrupt:
            print('正在终止程序，请耐心等待 20s 左右……')
            generator.exit()
            main_thread.join()
            print('程序终止成功！')
            break