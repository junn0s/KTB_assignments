import os
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain.llms import LlamaCpp

llm1 = LlamaCpp(
    model_path="llama-3-8b.gguf",
    temperature=0.7,
    max_tokens=256,
    top_p=0.9
)

# 첫 번째 프롬프트 (질문 요약)
summary_prompt = PromptTemplate.from_template("질문을 간결하게 요약하세요: {query}")

# 두 번째 프롬프트 (요약된 질문을 바탕으로 분석)
analysis_prompt = PromptTemplate.from_template("이 질문의 핵심 개념을 설명하세요: {summary}")

# 세 번째 프롬프트 (최종 응답 생성)
final_prompt = PromptTemplate.from_template("이 정보를 바탕으로 사용자에게 적절한 답변을 제공하세요: {analysis}")

# 체인 정의
def summarize(query):
    return llm1.invoke(summary_prompt.format(query=query))

def analyze(summary):
    return llm1.invoke(analysis_prompt.format(summary=summary))

def generate_response(analysis):
    return llm1.invoke(final_prompt.format(analysis=analysis))

# 체인 실행 (RunnableLambda 사용)
chain = RunnableLambda(summarize) | RunnableLambda(analyze) | RunnableLambda(generate_response)

# 사용자 질문 실행
query = "LLaMA랑 GPT랑 뭐가 더 좋아요? 그리고 LLaMA는 일론 머스크가 만들었나요?"
response = chain.invoke(query)

# 결과 출력
print("\n[LLM을 여러 번 호출하여 생성한 최종 답변]:")
print(response)