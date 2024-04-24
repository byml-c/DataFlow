import os
import re
import json
import time
import numpy as np
from tqdm import tqdm
from threading import Thread


import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import BisectingKMeans, KMeans, DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity

from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from sentence_transformers import SentenceTransformer

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, PromptTemplate, MessagesPlaceholder

from base import log, invoke

class Filter:
    qa: list
    hq: list
    lq: list
    name: str
    cluster: list
    stop_words: list
    log = log(__name__)

    def __init__(self, name='filter'):
        self.qa = []
        self.hq = []
        self.lq = []
        self.cluster = []
        self.name = name
        self.stop_words = [
            line.strip() for line in open('./config/hit_stopwords.txt', 'r', encoding='utf-8').readlines()
        ]
    
    def __dict__(self):
        return {
            'qa': self.qa,
            'hq': self.hq,
            'lq': self.lq,
            'cluster': self.cluster
        }

    def load(self):
        '''加载文件'''
        json_data = json.load(open(f'./{self.name}.json', 'r', encoding='utf-8'))
        self.qa = json_data['qa']
        self.hq = json_data['hq']
        self.lq = json_data['lq']
        self.cluster = json_data['cluster']

    def save(self):
        '''保存文件'''
        json.dump(
            obj=self.__dict__(),
            fp=open(f'./{self.name}.json', 'w', encoding='utf-8'),
            ensure_ascii=False
        )

    def load_file(self, path:str, save:bool=True):
        '''加载文件'''
        suffix = os.path.splitext(path)[1].lower().replace('.', '')
        if suffix == 'json':
            json_data = json.load(open(path, 'r', encoding='utf-8'))
            if 'qa' in json_data:
                for q in json_data['qa']:
                    q.update({'source': path.replace('-save.json', '')})
                    self.qa.append(q)
                self.log.info('从 {} 成功加载 {} 条QA对！'.format(path, len(json_data['qa'])))
        if save:
            self.save()
    
    def load_folder(self, input_root):
        '''递归处理文件夹内所有文件'''
        self.log.info('开始从文件夹 {} 加载QA对！'.format(input_root))
        for root, dirs, files in os.walk(input_root):
            for file in files:
                self.load_file(os.path.join(root, file), False)
        self.save()
        self.log.info('从文件夹 {} 加载完成！共加载 {} 条QA对！'.format(input_root, len(self.qa)))
    
    def filter(self, retry:bool=False):
        '''使用模式和模型进行预筛选'''
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
'''你将收到若干个QA对，格式为“问题：xxx\n回答：xxx\n\n”。
请根据以下标准评估每个QA对的质量，并将其标记为“优”或“劣”：

- 当回答未能充分解决问题，提供的信息不完整或不足以帮助用户解决问题时，应标记该QA对为“劣”
- 若模型给出的回答过于笼统，基于提供的信息不足而只能提供一种通用而非针对性的解决方案（例如，“在大学里遇到这类问题通常可以咨询教务处”），此类情况也应标记为“劣”。
- 若问题或回答中涉及具体的、独一无二的个人信息，造成其他用户无法从中获得有效参考（例如，“翠翠是学长还是学姐？”），则此QA对应标记为“劣”。

请确保输出结果为一个列表，其中包含了与输入QA对数量相同的元素，每个元素分别对应一个“优”或“劣”的评价。
'''
            ),
            HumanMessagePromptTemplate.from_template(
                '现在，请开始你的工作！\n\n{QA_pair}'
            )
        ])
        
        def check(item:dict):
            '''使用模式匹配的方式进行初步筛选，返回 True 表示通过筛选，False 表示未通过筛选'''
            if 'user_' in item['Q'] or 'user_' in item['A']:
                return False
            return True

        def run(source:list):
            block_size = 10
            for i in tqdm(range(0, len(source), block_size)):
                QA_pair = ''
                QA_source = source[i:i+block_size]
                for item in QA_source:
                    QA_pair += f'''问题: {item['Q']}\n回答: {item['A']}\n\n'''
                
                while True:
                    response = invoke(prompt, {'QA_pair': QA_pair})
                    # self.log.info(f'{QA_pair} {response} \n{len(QA_source)}')
                    try:
                        res = json.loads(response.strip().replace('\'', '\"'))
                        if len(res) != len(QA_source):
                            raise ValueError(f'返回结果长度错误！需要 {len(QA_source)}，返回 {len(res)}')
                        for i in range(len(res)):
                            if res[i] not in ['优', '劣']:
                                raise ValueError('返回结果错误！')
                        for i in range(len(res)):
                            if res[i] == '优':
                                self.hq.append(QA_source[i])
                            else:
                                self.lq.append(QA_source[i])
                        break
                    except Exception as e:
                        self.log.info(f'解析失败，报错：{e}')
        
        if not retry:
            self.hq = []
        temp = self.lq if retry else self.qa
        self.lq, qa_source = [], []
        for item in temp:
            if check(item):
                qa_source.append(item)
            else:
                self.lq.append(item)
        
        threads = []
        thread_num = 5
        thread_blocks_size = len(qa_source) // thread_num
        if thread_blocks_size == 0:
            thread_blocks_size = 1
        for i in range(0, len(qa_source), thread_blocks_size):
            t = Thread(target=run, args=([qa_source[i:i+thread_blocks_size]]))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        
        self.save()
    
    def embedding_vectorizer(self, sentences:list[str]):
        '''使用预训练的嵌入模型，将输入的句子向量化，返回 np.array 形式的句向量'''
        model_path = 'D:/Data/Models/m3e-base'
        m3e = SentenceTransformer(model_path)
        outpus = m3e.encode(sentences)
        return outpus
        # model_path = 'D:/Data/Models/nlp_gte_sentence-embedding_chinese-base'
        # pipeline_se = pipeline(
        #     Tasks.sentence_embedding,
        #     model=model_path,
        #     sequence_length=512
        # ) # sequence_length 代表最大文本长度，默认值为128
        # inputs = {
        #     'source_sentence': sentences
        # }
        # outputs = pipeline_se(input=inputs)
        # return outputs['text_embedding']
    
    def create_sentence_matrix(self, source:list):
        '''
            针对给定的 QA 对象输入，返回句向量矩阵
        '''
        def tokenize_by_jieba(item):
            words = jieba.lcut(item['Q'])
            return [word for word in words if word not in self.stop_words]
        def tokenize_by_llm(item):
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
'''
你的任务是从一个 QA 对中提取关键词。

工作流程如下：
- 你将会收到一个 QA 对，格式为“问题：xxx\n回答：xxx”。
- 你需要理解QA对的内容，对其进行概括，然后提取出有关提问的关键词。
- 接着，你需要将相似的关键词进行合并。

工作要求如下：
- 关键词应当是词语，尽量简短。
- 关键词之间应当尽量无关，且不能重复。
- 关键词无需太多，一般 3-5 个即可。

输出格式如下：
["关键词", "关键词", ... , "关键词"]

现在，请开始你的工作！
'''
                ),
                HumanMessagePromptTemplate.from_template(
                    '问题：{Q}\n回答：{A}'
                )
            ])
            while True:
                response = invoke(prompt, {'Q': item['Q'], 'A': item['A']}, False)
                try:
                    res = json.loads(response.strip().replace('\'', '\"'))
                    print(res)
                    return res
                except Exception as e:
                    self.log.info(f'解析失败，报错：{e}')
                
        questions = []
        for item_id in tqdm(range(len(source))):
            keys = []
            item = source[item_id]
            if len(item['keywords']) == 0:
                # keys = tokenize_by_llm(item)
                keys = tokenize_by_jieba(item)
            else:
                keys = item['keywords']

            keys = sorted([i.strip().lower() for i in keys])
            questions.append(' '.join(keys))

        # 使用词嵌入模型转换句向量
        sen_vec_matrix = self.embedding_vectorizer(questions)
        # 使用余弦相似度计算句向量之间的相似度
        cos_similarity_matrix = 1-cosine_similarity(sen_vec_matrix)
        cos_similarity_matrix[cos_similarity_matrix < 0] = 0
        # 使用 MDS 进行降维
        mds = MDS(n_components=2, dissimilarity='precomputed')
        
        # 使用 PCA 进行降维，将句向量降为 2 维
        # pca = PCA(n_components=2)
        # pca_weights = pca.fit_transform(sen_vec_matrix)
        res = mds.fit_transform(cos_similarity_matrix)
        print(res)
        return res

    def Bikmeans_cluster(self, source:list, recurse:bool=False):
        '''将给定的 QA 进行聚类'''
        # 转换成句向量
        sen_vec_matrix = self.create_sentence_matrix(source)

        # 确定最佳聚类数
        best_score = -1
        best_k = 0
        for k in tqdm(range(2, max(3, len(source)//10))):  # 尝试不同的聚类数量
            kmeans = BisectingKMeans(n_clusters=k)
            kmeans.fit(sen_vec_matrix)
            silhouette_avg = silhouette_score(sen_vec_matrix, kmeans.labels_)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k

        # self.log.info('最佳聚类数为：{}'.format(best_k))

        # 应用聚类
        kmeans = BisectingKMeans(n_clusters=best_k)
        kmeans.fit(sen_vec_matrix)
        clusters = kmeans.labels_

        # 存储分类结果
        clustered_strings = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_strings:
                clustered_strings[cluster_id] = []
            clustered_strings[cluster_id].append(source[i])
        result_list = [i[1] for i in clustered_strings.items()]
        
        # 递归处理每个聚类，直到每个聚类的大小 ≤ 10 条消息
        if recurse:
            temp_list = []
            for res in result_list:
                if len(res) <= 10:
                    temp_list.append(res)
                else:
                    temp_list += self.Bikmeans_cluster(res, recurse)
            result_list = temp_list
        return result_list
    
    def DBSCAN_cluster(self, source:list, recurse:bool=False):
        '''
            使用 DBSCAN 算法计算聚类
        '''
        sen_vec_matrix = self.create_sentence_matrix(source)

        affinity_propagation = DBSCAN(eps=0.2, min_samples=2)
        affinity_propagation.fit(sen_vec_matrix)
        clusters = affinity_propagation.labels_

        # 存储分类结果
        clustered_strings = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_strings:
                clustered_strings[cluster_id] = []
            clustered_strings[cluster_id].append(source[i])
        result_list = [i[1] for i in clustered_strings.items()]
        self.log.info(f'聚类存储完成，元素个数：{len(source)}，聚类个数：{len(result_list)}')
        
        # 递归处理每个聚类，直到每个聚类的大小 ≤ 10 条消息
        if recurse:
            temp_list = []
            if len(result_list) == 1:
                temp_list = self.Bikmeans_cluster(result_list[0], True)
            else:
                for res in result_list:
                    if len(res) <= 10:
                        temp_list.append(res)
                    else:
                        temp_list += self.DBSCAN_cluster(res, recurse)
            result_list = temp_list
        return result_list

    def classify(self):
        '''对 QA 对进行聚类，仅聚类 HQ 类型的 QA 对'''
        if len(self.hq) == 0:
            self.log.warning('有效QA为空，请先运行 filter 函数获得有效QA！')
            return 
        
        result = self.DBSCAN_cluster(self.hq, True)
        # 输出聚类结果
        self.log.info(f'聚类总数：{len(result)}')
        for i in range(len(result)):
            item = result[i]
            self.log.info("\nCluster "+str(i))
            for it in item:
                self.log.info(f'''问题：{it['Q']}\n回答：{it['A']}\n分类：{it['keywords']}\n来源：{it['source']}\n''')
        self.cluster = result

        self.save()
        json.dump(
            obj=result,
            fp=open(f'./{self.name}-cluster.json', 'w', encoding='utf-8'),
            ensure_ascii=False
        )

if __name__ == "__main__":
    f = Filter('QA')
    # f.load()
    f.load_file('../QA/南哪QA-save.json')
    f.load_folder('../QA/QQ')
    f.load_folder('../QA/Documents/AAA需增添水印的新文件/校园网相关')
    # f.filter()
    # print(len(f.lq), len(f.hq))
    # f.load_folder('../QA/QQ')
    # f.load_folder('../QA/Documents/AAA需增添水印的新文件/校园网相关')
    
    # f.filter()
    f.hq = f.qa
    f.classify()
    # for i in f.lq:
    #     f.log.info(f'''问题：{i['Q']}\n回答：{i['A']}\n分类：{i['type']}\n来源：{i['source']}\n\n''')
    