import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A,B):
    return dot(A,B) / (norm(A) * norm(B))

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]

from sklearn.feature_extraction.text import CountVectorizer

vector = CountVectorizer().fit(docs)
docs_v = vector.transform(docs).toarray()

print('문서 1과 문서2의 유사도 :',cos_sim(docs_v[0], docs_v[1]))
print('문서 1과 문서3의 유사도 :',cos_sim(docs_v[1], docs_v[2]))

#------------------------------------------------------------------
# 유사도 이용 간단한 recommend system
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TFIDF_Recommend_Class():
  
  def __init__(self, dataloader_path):
    self.path = dataloader_path
    

  def dataloader(self):
    data = pd.read_csv(self.path, low_memory=False)
    # 메모리 감당안됨
    data = data[:20000]
    return data

  def preprocessing_data(self, data):

    print('overview 열의 결측값의 수:',data['overview'].isnull().sum())
    data['overview'] = data['overview'].fillna('')

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['overview'])
    print('TF-IDF 행렬의 크기(shape) :',tfidf_matrix.shape)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print('코사인 유사도 연산 결과 :',cosine_sim.shape)

    title_to_index = dict(zip(data['title'], data.index))
    return cosine_sim, title_to_index

  def get_recommendations(self, title, cosine_sim, title_to_index):
    # 선택한 영화의 타이틀로부터 해당 영화의 인덱스를 받아온다.
    idx = title_to_index[title]

    # 해당 영화와 모든 영화와의 유사도를 가져온다.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬한다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # 가장 유사한 10개의 영화를 받아온다.
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스를 얻는다.
    movie_indices = [idx[0] for idx in sim_scores]

    # 가장 유사한 10개의 영화의 제목을 리턴한다.
    return data['title'].iloc[movie_indices]

path = "C:/Users/sonst/OneDrive/pytorch study/nlp/vector similarity/movies_metadata.csv"
tfidf_recommend = TFIDF_Recommend_Class(path)
data = tfidf_recommend.dataloader()
cosine_sim, title_to_index = tfidf_recommend.preprocessing_data(data)
print(tfidf_recommend.get_recommendations('Batman Forever', cosine_sim, title_to_index))

