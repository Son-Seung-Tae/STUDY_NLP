import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras


# 패딩이란? 문자열의 길이를 맞춰주는 행위

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
vocab_size = 5
tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size+2, oov_token="OOV")
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)


max_len = max(len(item) for item in encoded)
print(max_len)

for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
print(padded_np)


#----------------------------------------------------------------------
padded = keras.preprocessing.sequence.pad_sequences(encoded)
print(padded)

# value 값이 변하지 않는 이유를 모르겠슴..
k = 7
padded = keras.preprocessing.sequence.pad_sequences(encoded, padding="post", value=k)
print(padded)

