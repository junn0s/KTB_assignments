import pandas as pd

df = pd.DataFrame({"설명": ["빅데이터 분석", "데이터 과학", "머신 러닝", "딥 러닝"]})
df["약어"] = df["설명"].str.split().apply(lambda words: "".join(word[0] for word in words))
print(df)