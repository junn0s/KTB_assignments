import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)

news_article = input("뉴스 기사 전문을 입력하세요: ")

PROMPT = '''
너는 사용자에게 받은 뉴스 기사를 간결하고 명료하게 요약해주는 전문적인 브리핑 챗봇이야.
- 중요한 키포인트와 핵심 내용을 중심으로 최대한 간단명료하게 4줄 이내로 정리해줘.
- 맥락이 유지되도록 요약하되, 너무 단순 나열식으로 바꾸지 말고 자연스럽게 이어지도록 해줘.
- 오탈자나 비속어 없이 깔끔한 문체로 작성해줘.

[생성된 브리핑 요약]
'''

instruction = f"다음 기사 내용을 요약해줘:\n{news_article}"

messages = [
    {"role": "system", "content": f"{PROMPT}"},
    {"role": "user", "content": f"{instruction}"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

generation_kwargs = {
    "max_new_tokens": 512,
    "top_p": 0.9,
    "temperature": 0.6,
    "do_sample": True
}

generated_ids = model.generate(prompt, **generation_kwargs)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# 후처리 (뉴스 내용만 깔끔하게 추출)
if "assistant" in generated_text:
    generated_text = generated_text.split("assistant")[-1].strip()


sentences = re.split(r'(?<=[.!?])\s+', generated_text)

print("\n브리핑 내용:")
for sentence in sentences:
    print(sentence)