from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt
import nltk

# nltk.download('stopwords')
# nltk.download('punkt')

# 영어 불용어
# stop_words_list = stopwords.words('english')
# 한국어 불용어
# https://www.ranks.nl/stopwords/korean

# stopword 제거 함수 english
def stopword_remove(text):
    stop_words = set(stopwords.words('english')) 
    word_tokens = word_tokenize(text)

    result = []
    for word in word_tokens:
        if word not in stop_words: result.append(word)

    return result

# stopword 제거 함수 korean
def stopword_remove_kor(text):
    okt = Okt()
    stop_words = "은 는 이 가 나 너 우리 아 어 나 에 으로 로 에게 저 때는 게 다 안"
    stop_words = stop_words.split(" ")
    word_tokens = okt.morphs(text)

    result = [word for  word in word_tokens if word not in stop_words]
    
    return result



example = "Family is not an important thing. It's everything."
example_kor = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
print(word_tokenize(example))
print(stopword_remove(example))
print(Okt().morphs(example_kor))
print(stopword_remove_kor(example_kor))