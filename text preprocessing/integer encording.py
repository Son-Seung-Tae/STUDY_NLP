import collections
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words("english"))

sentences = sent_tokenize(raw_text)
print(sentences)

for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)
    result = []

    for word in tokenized_sentence:
        word = word.lower()

        if word not in stop_words:
            if len(word) > 2:
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0
                vocab[word] += 1
    preprocessed_sentences.append(result)
print(preprocessed_sentences)
print(vocab)

# vocab 정렬
vocab_sorted = sorted(vocab.items(), key = lambda x:x[1], reverse = True)
print(vocab_sorted)

# 빈도수가 1인 경우 제외하고 아닌경우 vacab이 큰 순서대로 작은 숫자 버여
word_to_idx = {}
i = 0
for (word, frequency) in vocab_sorted:
    if frequency == 1:
        break
    i += 1
    word_to_idx[word] = i
print(word_to_idx)
# 상위 5개 자르기
vocab_size = 5

# 인덱스가 5 초과인 단어 제거
words_frequency = [word for word, index in word_to_idx.items() if index >= vocab_size + 1]

# 해당 단어에 대한 인덱스 정보를 삭제
for w in words_frequency:
    del word_to_idx[w]
print(word_to_idx)

# vocab에 없는 단어를 처리해줄 OOV 생성
word_to_idx['OOV'] = len(word_to_idx) + 1
print(word_to_idx)

# 정수 인코딩
encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    for word in sentence:
        try:
            encoded_sentence.append(word_to_idx[word])
        except:
            encoded_sentence.append(word_to_idx["OOV"])
    encoded_sentences.append(encoded_sentence)

print(encoded_sentences)

# ------------------------------------------------------------------------
# 더 쉬운 방법 Counter를 이용하는 방법

from collections import Counter
print(preprocessed_sentences)
all_words_list = sum(preprocessed_sentences, [])
print(all_words_list)

vocab = Counter(all_words_list)
print(vocab)

vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장
print(vocab)

