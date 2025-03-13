# 사용자의 상황 제공 시 이메일 제작하는 챗봇

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "kakaocorp/kanana-nano-2.1b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir = "./models",
    torch_dtype=torch.bfloat16,        
    trust_remote_code=True,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./models")

subject = input("이메일 제목: ")
recipient = input("수신자 이메일 주소: ")
body = input("간략한 내용(상황), 수신자 이름, 발신자 이름: ")

PROMPT = '''너는 사용자 입력에 기반한 이메일을 생성하는 챗봇이야.
이메일에는 다음과 같은 내용이 무조건 포함되어야 해:
    - 인사말과 발신자 이름, 사용자 입력 기반한 구체적 본문 내용, 마무리 인사
    - 예의바르고 정중한 어조를 사용

[생성된 이메일]
'''

instruction = f'''이 정보들을 바탕으로 이메일을 생성해줘:
    - 제목: {subject}
    - 수신자: {recipient}
    - 본문: {body}
'''

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
    "max_new_tokens": 768,
    "top_p": 0.9,
    "temperature": 0.6,
    "do_sample": True
}

generated_ids = model.generate(prompt, **generation_kwargs)
generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if "assistant" in generated_text:
    generated_text = generated_text.split("assistant")[-1].strip()

print(generated_text)
