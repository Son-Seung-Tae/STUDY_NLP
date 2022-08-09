from konlpy.tag import Okt
import kss

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."

text2 = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
okt = Okt()
print(okt.morphs(text))
print(okt.morphs(text2))

# sentense tokenizer 사용 X 메모리 터짐
# print('한국어 문장 토큰화 :',kss.split_sentences(text2))
