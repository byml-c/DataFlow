import os
import re
import json
import time
import requests
from tqdm import tqdm
from threading import Thread

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, MessagesPlaceholder

from base import log, invoke

class MessageHandler:
    qa: list
    error: list
    blocks: list
    chains: list
    messages: list
    user_map: dict
    user_num: int
    online: bool
    categories: list
    log = log(__name__)

    def __init__(self):
        self.qa = []
        self.error = []
        self.blocks = []
        self.online = True
        self.messages = []
        self.chains = []
        self.user_num = 0
        self.user_map = {}

        config = json.load(open('./config/qa.json', 'r', encoding='utf-8'))
        self.categories = config['categories']
    
    def print(self, obj, output=None, addition=''):
        if output is not None:
            with open(output, 'a', encoding='utf-8') as f:
                f.write('[{}] {}\n'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime()), addition))
                f.write(obj.__str__())
                f.write('\n')
        else:
            print(obj)

    def print_msg(self, msg_list, output=None):
        if output is not None:
            with open(output, 'a', encoding='utf-8') as f:
                f.write('[{}]\n'.format(time.strftime(r'%Y-%m-%d %H:%M:%S', time.localtime())))
                for msg in msg_list:
                    f.write('[{}]({}): {}\n'.format(msg['time'], msg['user'], msg['message']))
                f.write('\n')
        else:
            for msg in msg_list:
                print('[{}]({}): {}'.format(msg['time'], msg['user'], msg['message']))
            print('\n')
            
    def __dict__(self):
        return {
            'qa': self.qa,
            'error': self.error,
            'blocks': self.blocks,
            'chains': self.chains,
            'messages': self.messages,
            'user_map': self.user_map
        }

    def save(self, output_path:str):
        '''保存聊天记录'''
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     for m in self.messages:
        #         f.write('[{}]({})\n{}\n\n'.format(m['time'], m['user'], m['message']))
        json_path = os.path.splitext(output_path)[0]+'-save.json'
        json.dump(
            obj=self.__dict__(),
            fp=open(json_path, 'w', encoding='utf-8'),
            ensure_ascii=False
        )
    
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

    def create_time_block(self):
        '''创建根据时间和 @ 关系创建消息块'''
        self.blocks = []

        # 根据时间和消息链，划分消息块
        max_block_size = 1024
        def append_time_block(time_chain_temp:list):
            if len(time_chain_temp) == 0:
                return 
            time_messages_id_temp = []
            for chain in time_chain_temp:
                time_messages_id_temp += chain
            time_messages_id_temp.sort()
            self.blocks.append(time_messages_id_temp)
        def get_length(chain):
            count = 0
            for c in chain:
                count += self.messages[c]['length']
            return count
        def split_chains(chains:list, duration:int=60*60):
            '''
                依据时间粒度、@ 关系，划分在最大长度以内的消息块
                划分原则：时间粒度优先，保留 @ 关系。
                    但是对于时间粒度 < 225s 的块（即二分粒度为 125s 的块），采取按照字符强制划分的方式。
            '''
            chain_block_temp, chain_size = [], 0
            last_time = self.messages[chains[0][0]]['stamp']
            for chain in chains:
                if duration >= 225:
                    stamp = self.messages[chain[0]]['stamp']
                    if stamp - last_time <= duration:
                        chain_block_temp.append(chain)
                        chain_size += get_length(chain)
                    else:
                        if chain_size > max_block_size:
                            split_chains(chain_block_temp, duration//2)
                        else:
                            append_time_block(chain_block_temp)
                        chain_block_temp, chain_size = [chain], get_length(chain)
                    last_time = stamp
                else:
                    if chain_size + get_length(chain) > max_block_size:
                        append_time_block(chain_block_temp)
                        chain_block_temp, chain_size = [chain], get_length(chain)
                    else:
                        chain_block_temp.append(chain)
                        chain_size += get_length(chain)
            if chain_size > 0:
                if duration >= 225 and chain_size > max_block_size:
                    split_chains(chain_block_temp, duration//2)
                else:
                    append_time_block(chain_block_temp)
        split_chains(self.chains)
    
    def generate_QA_by_time_block(self, model:str=None):
        '''然后由 AI 根据消息块总结QA'''
        self.create_time_block()
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
'''
- Role: 聊天记录分析专家
- Background: 您将接收一组以特定格式编写的聊天记录，并从中提取热点话题，然后生成相关的问答对。
- Profile: 您是一位专业的聊天记录分析专家，具备深厚的文本分析能力和话题提取技巧。
- Skills: 您需要运用文本解析、话题识别、问题生成和答案撰写等技能。
- Goals: 您需要从聊天记录中识别热点话题，并围绕每个话题生成2-3个相关的问答对。
- Constrains: 
    - 您生成的问题应具有普遍性，能够覆盖话题的主要方面，答案应详细且准确，格式需符合指定的JSON结构。
    - 如果提供的信息不足以生成QA，请输出“无答案”三个字。
- OutputFormat: 您需要按照以下JSON格式输出结果，确保每个问答对都包含“问题”、“回答”、“分类”、“关键词”和“依据”。
```json
[
    {{
        "问题": "xxx",
        "回答": "xxx",
        "分类": "xxx",
        "关键词": [
            "xxx", ... ,
            "xxx"
        ],
        "依据": [
            "xxx", ... ,
            "xxx"
        ]
    }}, ... ,
    {{
        "问题": "xxx",
        "回答": "xxx",
        "分类": "xxx",
        "关键词": [
            "xxx", ... ,
            "xxx"
        ],
        "依据": [
            "xxx", ... ,
            "xxx"
        ]
    }}
]
```
- OutputRequirement:
    - 问题、回答：问答对文本
    - 分类：所属分类，应为这些分类中的一个：{categories}
    - 关键词：根据提问生成关键词，用于检索
- Workflow:
    1. 仔细阅读提供的聊天记录。
    2. 识别记录中的热点话题。
    3. 为每个热点话题生成2-3个相关问答对。
    4. 确保问题和答案与原文相关，风格自然。
    5. 使用指定的格式输出问答对。
- Examples: 
    - 聊天记录片段：
        [2023.8.29 21:52:04](user_0001): 请问我们体育课有没有必须选（过）的课？
        [2023.8.29 21:52:04](user_0002): @user_0001 这是啥意思？
        [2023.8.29 21:52:48](user_0002): 体育课都必须过，不过没学分毕业证拿不了啊。
        [2023.8.29 22:16:38](user_0003): 各门课应该选几节啊
        [2023.8.29 22:16:59](user_0004): @user_0003 你可以选到最多，以免抽不中
        [2023.8.29 22:17:03](user_0004): 多中可以退
    - QA: 
        [{{
            "问题": "我们体育课有没有必须选（过）的课？",
            "回答": "体育课都必须过，否则没有对应的学分，就会拿不了毕业证。",
            "分类": "学习资源与课程通知",
            "关键词": ["体育课", "必须过", "学分", "毕业证"],
            "依据": [
                "[2023.8.29 21:52:04](user_0001): 请问我们体育课有没有必须选（过）的课？",
                "[2023.8.29 21:52:48](user_0002): 体育课都必须过，不过没学分毕业证拿不了啊。"
            ]
        }}, {{
            "问题": "各门课应该选几节啊",
            "回答": "你可以选到选课上限，以避免抽签时出现抽不中的情况，抽中了多门课可以退。",
            "分类": "学习资源与课程通知",
            "关键词": ["各门课", "选几节", "多选", "退课"],
            "依据": [
                "[2023.8.29 22:16:38](user_0003): 各门课应该选几节啊",
                "[2023.8.29 22:17:03](user_0004): 多中可以退"
            ]
        }}]
'''
            ), HumanMessagePromptTemplate.from_template(
'''聊天记录片段：\n{messages}\n\n根据聊天记录内容，严格按照输出格式，我的输出如下：\n'''
            )
        ])

        def run(block:list):
            if len(block) <= 1:
                self.error.append(block)
                return
            
            messages = ''
            for j in block:
                msg = self.messages[j]
                messages += '[{}]({}): {}\n'.format(msg['time'], msg['user'], msg['message'])
            while True:
                response = invoke(prompt, {
                    'messages': messages,
                    'categories': '、'.join(self.categories)
                }, model)

                if response == '':
                    time.sleep(0.5)
                    continue
                if response == '<ERROR>':
                    self.error.append(block)
                    self.log.log('模型返回错误，加入错误列表！数据：\n{}'.format(block), 'E')
                    break
                if re.search('无答案', response):
                    self.log.log(f'无答案，加入错误列表！数据：\n{block}\n返回：\n{response}', 'E')
                    self.error.append(block)
                    break
                
                try:
                    res = re.search(r'```json((.|[\n\r])*)```', response, re.S)
                    res = json.loads(res.group(1) if res else response)
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
                    self.log.log('出错：{}，模型返回：{}'.format(err, response), 'E')
                    time.sleep(1)

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
                threads = [t for t in threads if t.is_alive()]
        
        for t in threads:
            t.join()

    def read(self, path:str):
        '''从文件加载聊天记录'''
        path = os.path.splitext(path)[0]+'-save.json'
        self_obj = json.load(open(path, 'r', encoding='utf-8'))

        self.qa = self_obj['qa']
        self.error = self_obj['error']
        self.blocks = self_obj['blocks']
        self.chains = self_obj['chains']
        self.messages = self_obj['messages']
        self.user_map = self_obj['user_map']
        self.user_num = self.user_map.__len__()

    def initialize(self, path:str):
        '''从源文件中读取聊天记录并进行初步过滤和脱敏处理'''
        self.log.log('初始化处理聊天记录，路径：{}'.format(path))
        self.load(path)
        self.log.log('聊天记录导入完成，共计{}条消息'.format(len(self.messages)))
        self.log.log('正在隐藏用户信息和过滤消息')
        self.process()
        self.log.log('用户信息隐藏和消息过滤完成，共计{}位用户，{}条消息' \
                 .format(self.user_num, len(self.messages)))

    def load(self, path:str):
        '''加载聊天记录'''
        self.messages, start = [], False
        message_list = open(path, 'r', encoding='utf-8').readlines()
        for line in message_list:
            line = line.strip().strip('\uFEFF')
            s = re.search(r'(^\d{4}-\d{1,2}-\d{1,2}\s\d{1,2}:\d{1,2}:\d{1,2})\s?(.*)$', line)
            if s is not None:
                start = True
                user_data = re.search(r'(.*)[(<](.*?)[)>]$', \
                                      s.group(2).strip())
                nickname = re.sub(r'【.{,5}】', '', user_data.group(1)).strip() \
                    if user_data is not None else ''
                QQ_id = user_data.group(2).strip() \
                    if user_data is not None else s.group(2).strip()
                self.messages.append({
                    'user': QQ_id,
                    'message': '',
                    'length': 0,
                    'nick': nickname,
                    'time': s.group(1),
                    'stamp': time.mktime(time.strptime(s.group(1), r'%Y-%m-%d %H:%M:%S'))
                })
            elif not line == '' and start:
                self.messages[-1]['message'] += line.strip()+'\n'

    def process(self):
        '''处理消息，过滤无效消息，隐藏用户信息'''
        def filter(m):
            '''判断消息是否应该被过保留'''
            filter_sentences = [
                '大家好，我是', '加入了本群', '加入本群', '修改了群名称', '点击添加好友', 
                '请使用最新版手机QQ体验新功能', '欢迎欢迎～进群就是一家人了', 
                '撤回了一条消息'
            ]
            if m['nick'] == '系统消息':
                return None
            msg = m['message'].strip()
            msg = re.sub(r'\[[^\]]*?\]', ' ', msg).strip()
            if msg == '':
                return None
            for sentence in filter_sentences:
                if sentence in msg:
                    return None
            m['message'] = msg
            return m

        '''隐藏用户信息'''
        nick_map = {}
        self.user_map = {}
        self.user_num, max_length = 0, 0
        
        # 为所有用户建立映射
        for m in self.messages:
            nick_map[m['nick']] = m['user']
            if len(m['nick']) > max_length:
                max_length = len(m['nick'])
            if m['user'] not in self.user_map:
                self.user_num += 1
                self.user_map[m['user']] = {
                    'id': 'user_{:04d}'.format(self.user_num),
                    'messages': [], 'at': [], 'nick': [m['nick']]
                }
            else:
                if m['nick'] not in self.user_map[m['user']]['nick']:
                    self.user_map[m['user']]['nick'].append(m['nick'])

        message_temp, temp_len = [], 0
        for i in range(len(self.messages)):
            # 过滤消息
            msg = filter(self.messages[i])
            if msg is None:
                continue

            # 替换用户名
            user_name = msg['user']
            msg['user'] = self.user_map[user_name]['id']

            # 替换内容中的 @ 引用
            content = msg['message']
            content_len = len(content)
            j, chain_find = 0, False
            while j < content_len:
                if content[j] == '@':
                    user_temp, find_user = '', None
                    for k in range(j+1, min(j+1+max_length, content_len)):
                        user_temp += content[k]
                        if nick_map.get(user_temp) is not None:
                            find_user = user_temp
                    if find_user is not None:
                        j += len(find_user)
                        user_QQ_id = nick_map[find_user]
                        sub_name = self.user_map[user_QQ_id]['id']
                        if len(self.user_map[user_QQ_id]['at']) == 0 \
                            or self.user_map[user_QQ_id]['at'][-1] != i:
                            self.user_map[user_QQ_id]['at'].append(temp_len)

                            # 建立 @ 链
                            if not chain_find:
                                chain_i, chain_len = -1, len(self.chains)
                                while chain_i >= -chain_len:
                                    # 如果在两个小时后 @，认为没有意义，不放在同一个链中。
                                    if msg['stamp'] - self.messages[self.chains[chain_i][-1]]['stamp'] > 2*60*60:
                                        break
                                    for chain_j in self.chains[chain_i]:
                                        if message_temp[chain_j]['user'] == sub_name:
                                            chain_find = True
                                            self.chains[chain_i].append(temp_len)
                                            break
                                    if chain_find:
                                        break
                                    chain_i -= 1
                        msg['message'] = msg['message'].replace(find_user, sub_name)
                j += 1
            
            # 统计消息长度
            msg['length'] = len(msg['message'])
            # 添加消息到所属用户和消息列表
            if not chain_find:
                self.chains.append([temp_len])
            message_temp.append(msg)
            self.user_map[user_name]['messages'].append(temp_len)
            temp_len += 1
        self.messages = message_temp
    
    def handle(self, input_path:str, output_path:str='./output.txt', init:bool=None, model:str=None):
        '''调用此函数一键处理消息文件'''
        if init is None:
            init = not os.path.exists(output_path)
            
        if init:
            self.initialize(input_path)
            self.save(output_path)
        else:
            self.read(output_path)

        if init or len(self.qa) == 0:
            self.log.log('正在生成QA对')
            self.generate_QA_by_time_block(model)
            self.save(output_path)
            self.log.log('QA对生成完成，生成QA对：{}个'.format(len(self.qa)))
        self.log.log('{} 处理完成，输出至：{}'.format(input_path, output_path))

if __name__ == '__main__':
    data_root = '../RawData/'
    output_root = '../QA/'

    handle_list = ['QQ/南哪23级本科①群.txt', 'QQ/南哪2022级咨询交流①群.txt', 'QQ/南哪2021级咨询交流群.txt']

    for file_name in handle_list:
        h = MessageHandler()
        h.handle(data_root+file_name, output_root+file_name)

    # handlers = []
    # tot = 0
    # for files in os.listdir('./QQ'):
    #     if files.endswith('.txt'):
    #         h = MessageHandler()
    #         h.handle('./QQ/'+files, './QQ_save/'+files, True)
    #         handlers.append(h)
    #         tot += len(h.messages)
    # print(tot)