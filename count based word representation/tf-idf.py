# TF-IDF
# TF => 특정 문서 d에서 단어 t의 등장 횟수
# DF => 특정 단어 t가 등장한 문서의 수
# IDF는 DF의 역수이나 log를 취함 log를 취하지 않으면 매우 큰 가중치가 부여되게 됨
# idf = log(n /(1 + df(t)) )
import pandas as pd
from math import log

class TFIDFClass():
    def __init__(self, docs):
        self.docs = docs
        self.vocab = list(set(w for doc in docs for w in doc.split()))
        self.N = len(docs)

    def tf(self,t,d):
        return d.count(t)

    def idf(self, t):
        df = 0
        for doc in self.docs:
            df += t in doc
        return log(self.N/(df + 1))

    def tfidf(self, t,d):
        return self.tf(t,d)*self.idf(t)

    def build_tfidf(self):
        result = []
        for doc in self.docs:
            result.append([])
            for i in range(len(self.vocab)):
                t = self.vocab[i]
                result[-1].append(self.tfidf(t, doc))

        
        tf_idf = pd.DataFrame(result, columns=self.vocab)
        return tf_idf
                

docs = [
  '먹고 싶은 사과',
  '먹고 싶은 바나나',
  '길고 노란 바나나 바나나',
  '저는 과일이 좋아요'
]

tf_c = TFIDFClass(docs=docs)
print(tf_c.build_tfidf())

#-------------------------------------------------------------
# 라이브러리 활용 TF-IDF
from sklearn.feature_extraction.text import CountVectorizer

corpus = docs
vector = CountVectorizer()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidfv = TfidfVectorizer().fit(corpus)
sort_value = sorted(tfidfv.vocabulary_.items(), key = lambda item: item[1])
print(pd.DataFrame(tfidfv.transform(corpus).toarray(), columns=dict(sort_value)))
print(tfidfv.vocabulary_)
