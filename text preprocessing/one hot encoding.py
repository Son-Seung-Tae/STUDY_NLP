from tensorflow import keras

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([text])
print('단어 집합 :',tokenizer.word_index)

sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
print(encoded)

one_hot = keras.utils.to_categorical(encoded)
print(one_hot)

#-------------------------------------------------------
#torch ver
import torch
import torch.nn.functional as F
print(F.one_hot(torch.tensor(encoded)))