import os
import json
import time
import numpy as np
from tqdm import tqdm


import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import BisectingKMeans, KMeans

# from modelscope.models import Model
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
from sentence_transformers import SentenceTransformer

from base import log, invoke

class Filter:
    qa: list
    stop_words: list
    log = log(__name__)

    def __init__(self):
        self.qa = []

        self.stop_words = [
            line.strip() for line in open('./config/hit_stopwords.txt', 'r', encoding='utf-8').readlines()
        ]
    
    def load(self, path:str):
        '''加载文件'''
        suffix = os.path.splitext(path)[1].lower().replace('.', '')
        if suffix == 'json':
            json_data = json.load(open(path, 'r', encoding='utf-8'))
            if 'qa' in json_data:
                for q in json_data['qa']:
                    q.update({'source': path.replace('-save.json', '')})
                    self.qa.append(q)
                self.log.info('从 {} 成功加载 {} 条QA对！'.format(path, len(json_data['qa'])))
    
    def load_folder(self, input_root):
        '''递归处理文件夹内所有文件'''
        self.log.info('开始从文件夹 {} 加载QA对！'.format(input_root))
        for root, dirs, files in os.walk(input_root):
            for file in files:
                self.load(os.path.join(root, file))
        self.log.info('从文件夹 {} 加载完成！共加载 {} 条QA对！'.format(input_root, len(self.qa)))
    
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

    def cluster(self, source:list, recurse:bool=False):
        '''将给定的 QA 进行聚类'''
        def tokenize(text):
            words = jieba.lcut(text)
            return [word for word in words if word not in self.stop_words]
        
        questions = []
        for item in source:
            keys = []
            if len(item['keywords']) == 0:
                keys = tokenize(item['Q'])
            else:
                keys = item['keywords']

            keys = sorted([i.strip().lower() for i in keys])
            questions.append(' '.join(keys))

        # 特征提取
        # tfidf_vectorizer = TfidfVectorizer()
        # sen_vec_matrix = tfidf_vectorizer.fit_transform(questions)
        sen_vec_matrix = self.embedding_vectorizer(questions)

        # 确定最佳聚类数
        best_score = -1
        best_k = 0
        for k in tqdm(range(2, max(3, len(questions)//2))):  # 尝试不同的聚类数量
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
                    temp_list += self.cluster(res, recurse)
            result_list = temp_list
        return result_list
    
    def classify(self):
        result = self.cluster(self.qa, True)
        # 输出聚类结果
        self.log.info(f'聚类总数：{len(result)}')
        for i in range(len(result)):
            item = result[i]
            self.log.info("\nCluster "+str(i))
            for it in item:
                self.log.info(f'''问题：{it['Q']}\n回答：{it['A']}\n分类：{it['type']}\n来源：{it['source']}\n''')

if __name__ == "__main__":
    f = Filter()
    f.load_folder('../QA/QQ')
    f.load_folder('../QA/Documents/AAA需增添水印的新文件/校园网相关')
    # f.load('../QA/南哪QA-save.json')
    f.classify()
    # f.embedding_vectorizer(['你好，这是一段测试', '测试中'])