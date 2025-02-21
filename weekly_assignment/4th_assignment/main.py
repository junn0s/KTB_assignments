from typing import Union
from fastapi import FastAPI
import model  

logic_model = model.LogicGateAI(num_epochs=1000, lr=0.01)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Logic Gate AI Server"}

# 예측 엔드포인트
# 예시 URL: /predict/left/1/right/0/gate/XOR
@app.get("/predict/left/{left}/right/{right}/gate/{gate}")
def predict(left: int, right: int, gate: str):
    gate = gate.upper()
    if gate not in ["AND", "OR", "XOR", "NOT"]:
        return {"error": "Invalid gate type. Use one of: AND, OR, XOR, NOT"}
    
    # NOT 연산은 단항 연산이므로 두 번째 입력은 무시(0으로 고정)
    if gate == "NOT":
        input_data = [left, 0] + logic_model.gate_types[gate]
    else:
        input_data = [left, right] + logic_model.gate_types[gate]
    
    # predict 함수는 하나의 샘플 리스트를 인자로 받으므로 [input_data]로 감싸서 전달
    prediction = logic_model.predict([input_data])
    return {"result": int(prediction.item())}

# 모델 학습 엔드포인트
@app.post("/train")
def train():
    logic_model.train()
    return {"result": "Model trained successfully"}