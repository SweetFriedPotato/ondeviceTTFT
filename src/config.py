# src/config.py

BACKEND = "vllm"
# 사용할 모델
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  

# TTFT 실험용 프롬프트
TTFT_PROMPT = "Hello, this is a TTFT measurement test."
MAX_NEW_TOKENS_FOR_TTFT = 1
NUM_TRIALS = 5             # TTFT 반복 횟수

# device 설정
DTYPE = "float32"          