# API key 발급 이슈로 sentence transformer 모델 사용
# SBERT 활용으로 높은 성능의 임베딩 

import torch
from sentence_transformers import SentenceTransformer, util

# SBERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2') 

sentences = [
    "Milo는 배가 고픕니다.",
    "판교 주변에 식당이 많습니다.",
    "판교 주변에 술집이 많습니다.",
    "Sandy는 배가 고픕니다."
]

embeddings = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.cos_sim(embeddings, embeddings)

print("임베딩 벡터 크기:", embeddings.shape)
print("첫 번째 문장의 임베딩 벡터:", embeddings[0])
print("각 문장과 가장 유사한 문장:")
for i in range(2):
    sim_scores = cosine_scores[i] 
    sim_scores[i] = -1  
    most_similar_idx = torch.argmax(sim_scores).item()  
    print(f"'{sentences[i]}' → '{sentences[most_similar_idx]}' (유사도: {cosine_scores[i][most_similar_idx]:.4f})")