# 정규 표현식
import re

r = re.compile("a.c")
print(r.search("kkk"))
print(r.search("abc"))

r = re.compile("^ab")
print(r.search("bbc"))
print(r.search("abz"))


# 정규 표현식과 매칭되는 문자열을 list로 return
text = """이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""
# /d 라는 숫자를 찾아서 list의 형태로 return
re.findall("\d+", text)

# re.sub 패턴과 일치하는 문자열을 다른 문자열로 치환
text = "Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."

preprocessed_text = re.sub('[^a-zA-Z]', ' ', text)
print(preprocessed_text)


text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""

# split으로 분할할떄에 + 를 붙임으로서 공백이 여러개여도 하나로 분할
print(re.split("\s", text))
print(re.split("\s+", text))

# 대문자만 찾기
print(re.findall("[A-Z]", text))
# 숫자만 찾기
print(re.findall("\d+", text))
# 연속해서 4번 등장하는 경우 찾기
print(re.findall("[A-Z][a-z]+", text))
