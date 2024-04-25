import sqlite3
import hashlib
import time
import json
import uuid
from base import log

class Database:
    name: str
    sheet: str
    log: log
    db_connect: sqlite3.Connection
    db_cursor: sqlite3.Cursor

    time_format = r'%Y-%m-%d %H:%M:%S'

    def __init__(self, name:str, sheet:str) -> None:
        self.name = name
        self.sheet = sheet
        self.log = log(name)
        # 连接数据库
        self.db_connect = sqlite3.connect(f'{name}.db', check_same_thread=False)
        self.db_cursor = self.db_connect.cursor()
        # 选中或创建表
        query = f'''SELECT name FROM sqlite_master WHERE type='table' AND name='{sheet}';'''
        self.db_cursor.execute(query)
        if self.db_cursor.fetchone() is None:
            query = f'''CREATE TABLE {sheet}
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
            self.log.log(f'数据库创建成功！')
        else:
            self.log.log(f'数据库连接成功！')
        
    def __del__(self):
        '''析构函数，关闭数据库连接'''
        self.db_connect.close()

    def insert(self, data:dict) -> None:
        '''将数据加入数据库'''
        id = uuid.uuid1()
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

        query = f'''INSERT INTO '{self.sheet}' (id, type, metadata, source, author, time) VALUES ("{id}", "{type}", "{metadata}", "{source}", "{author}", "{rtime}");'''
        self.db_cursor.execute(query)
        self.db_connect.commit()
    
    def fetchall(self)->list:
        '''
            返回表中的所有数据
        '''

        query = f'''SELECT * FROM {self.sheet};'''
        self.db_cursor.execute(query)
        return self.db_cursor.fetchall()