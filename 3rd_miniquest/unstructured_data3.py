import re
import pandas as pd

text = "자연어 처리는 재미있다. 파이썬과 pandas를 활용하면 편리하다. 데이터 분석은 흥미롭다."

sentences = text.split(". ")
word_counts = [len(sentence.split()) for sentence in sentences]
df = pd.DataFrame({
    '단어 수': word_counts
})

print(df)