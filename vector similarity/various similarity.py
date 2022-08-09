import numpy as np

# 유클리디안 거리
def euclidean_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

doc1 = np.array((2,3,0,1))
doc2 = np.array((1,2,3,1))
doc3 = np.array((2,1,2,2))
docQ = np.array((1,1,0,1))

print('문서1과 문서Q의 거리 :',euclidean_dist(doc1,docQ))
print('문서2과 문서Q의 거리 :',euclidean_dist(doc2,docQ))
print('문서3과 문서Q의 거리 :',euclidean_dist(doc3,docQ))

#자카드 유사도
# 두 문장의 교집합 / 합집합

doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"

tok_1 = doc1.split(" ")
tok_2 = doc2.split(" ")
print(tok_1)
print(tok_2)

union = set(tok_1).union(set(tok_2))
print('문서1과 문서2의 합집합 :',union)

intersection = set(tok_1).intersection(set(tok_2))
print('문서1과 문서2의 교집합 :',intersection)

print('자카드 유사도 :',len(intersection)/len(union))
