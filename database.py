import sqlite3
import hashlib
import time
import json
import uuid
from base import log

class Database:
    name: str
    sheet: str
    logger: log
    db_connect: sqlite3.Connection
    db_cursor: sqlite3.Cursor

    time_format = r'%Y-%m-%d %H:%M:%S'

    def __init__(self, name:str, sheet:str) -> None:
        self.name = name
        self.sheet = sheet
        self.logger = log(name)
        # 连接数据库
        self.db_connect = sqlite3.connect(f'{name}.db', check_same_thread=False)
        self.db_cursor = self.db_connect.cursor()
        # 选中或创建表
        query = f'''SELECT name FROM sqlite_master WHERE type='table' AND name='{sheet}';'''
        self.db_cursor.execute(query)
        if self.db_cursor.fetchone() is None:
            query = f'''CREATE TABLE '{sheet}'
            (
                id          TEXT        PRIMARY KEY,
                type        TEXT        ,
                metadata    JSON        NOT NULL,
                source      TEXT        NOT NULL,
                author      TEXT        NOT NULL,
                time        TEXT        NOT NULL
            );
            '''
            # data 包含：Q, A, keywords, refs
            self.db_cursor.execute(query)
            self.db_connect.commit()
            self.logger.log(f'数据库创建成功！')
        else:
            self.logger.log(f'数据库连接成功！')
        
    def __del__(self):
        '''析构函数，关闭数据库连接'''
        self.db_connect.close()

    def insert(self, data:dict) -> None:
        '''将数据加入数据库'''
        id = str(uuid.uuid1())
        type = data['type']
        metadata = json.dumps({
            'Q': data['Q'],
            'A': data['A'],
            'keywords': data['keywords'],
            'refs': data['refs']
        }, ensure_ascii=False)
        source = data['source']
        author = data['author']
        rtime = time.strftime(self.time_format, time.localtime())

        query = f'''INSERT INTO {self.sheet} (id, type, metadata, source, author, time) VALUES (?, ?, ?, ?, ?, ?);'''
        self.db_cursor.execute(query, (id, type, metadata, source, author, rtime))
        self.db_connect.commit()
    
    def fetchall(self)->list:
        '''
            返回表中的所有数据
        '''

        query = f'''SELECT * FROM '{self.sheet}';'''
        self.db_cursor.execute(query)
        return self.db_cursor.fetchall()

    def print(self):
        '''打印数据库中的所有数据'''
        data_list = self.fetchall()
        for data in data_list:
            meta = json.loads(data[2])
            print(f'id: {data[0]}')
            print(f'time: {data[5]}')
            print(f'type: {data[1]}')
            print(f'author: {data[4]}')
            print(f'source: {data[3]}')
            print(f'keywords: {meta["keywords"]}')
            print(f'Q: {meta["Q"]}')
            print(f'A: {meta["A"]}')
            print(f'refs: {meta["refs"]}')
            print('-'*10)