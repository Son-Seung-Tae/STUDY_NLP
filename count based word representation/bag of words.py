from konlpy.tag import Okt

class BowClass:

    def __init__(self, document):
        self.document = document
        self.okt = Okt()

    def build_bag_of_words(self):
        # 온점 제거 및 형태소 분석
        self.document = self.document.replace('.', '')
        tokenized_document = self.okt.morphs(self.document)

        word_to_index = {}
        bow = []

        for word in tokenized_document:  
            if word not in word_to_index.keys():
                word_to_index[word] = len(word_to_index)  
                # BoW에 전부 기본값 1을 넣는다.
                bow.insert(len(word_to_index) - 1, 1)
            else:
                # 재등장하는 단어의 인덱스
                index = word_to_index.get(word)
                # 재등장한 단어는 해당하는 인덱스의 위치에 1을 더한다.
                bow[index] = bow[index] + 1
        return word_to_index, bow

doc1 = "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다."
bow_c = BowClass(doc1)
w_idx, bow_v = bow_c.build_bag_of_words()
print(w_idx)
print(bow_v)

#----------------------------------------------------------------------
#countvextorizer를 이용한 bow 클래스 만들기

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()

print("bow_v", vector.fit_transform(corpus).toarray())
print("w_idx", vector.vocabulary_)


# 불용어를 제거한 Bow

vector = CountVectorizer(stop_words="english")
print("bow_v", vector.fit_transform(corpus).toarray())
print("w_idx", vector.vocabulary_)
