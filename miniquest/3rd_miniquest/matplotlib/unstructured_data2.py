import re

text = "안녕하세요!!! 저는 AI 모델-입니다. 12345 데이터를   정리해 보겠습니다."
clean_text = re.sub(r"[^가-힣\s]", "", text)
clean_text = re.sub(r"\s+", " ", clean_text).strip()
print(clean_text)
