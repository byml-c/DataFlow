import os
import json
import time
from tqdm import tqdm


import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import BisectingKMeans, KMeans

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
    
    def classify(self):
        def tokenize(text):
            words = jieba.lcut(text)
            return [word for word in words if word not in self.stop_words]
        # questions = [' '.join(tokenize(q['Q'])) for q in self.qa]
        questions = [
            ' '.join(
                q['keywords'] if len(q['keywords']) > 0 else tokenize(q['Q'])
            ) for q in self.qa
        ]

        # 特征提取
        tfidf_vectorizer = TfidfVectorizer(max_features=10000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(questions)

        # 确定最佳聚类数
        best_score = -1
        best_k = 0
        for k in tqdm(range(2, max(3, len(questions)))):  # 尝试不同的聚类数量
            kmeans = BisectingKMeans(n_clusters=k)
            kmeans.fit(tfidf_matrix)
            silhouette_avg = silhouette_score(tfidf_matrix, kmeans.labels_)
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_k = k

        self.log.info('最佳聚类数为：{}'.format(best_k))

        # 应用聚类
        kmeans = BisectingKMeans(n_clusters=best_k)
        kmeans.fit(tfidf_matrix)
        clusters = kmeans.labels_

        # 存储分类结果
        clustered_strings = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in clustered_strings:
                clustered_strings[cluster_id] = []
            clustered_strings[cluster_id].append(self.qa[i])
        
        # 输出聚类结果
        for cluster_id, cluster_strings in clustered_strings.items():
            self.log.info("\nCluster "+str(cluster_id))
            for string in cluster_strings:
                self.log.info(f'''问题：{string['Q']}\n回答：{string['A']}\n分类：{string['type']}\n来源：{string['source']}\n''')

if __name__ == "__main__":
    f = Filter()
    f.load_folder('../QA/QQ')
    f.load_folder('../QA/Documents/AAA需增添水印的新文件/校园网相关')
    f.load('../QA/南哪QA-save.json')
    f.classify()