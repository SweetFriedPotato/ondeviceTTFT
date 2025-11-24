# src/config.py

# 사용할 모델
#MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"   # 3B급 기본 모델
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# TTFT 실험용 프롬프트
TTFT_PROMPT = "Hello, this is a TTFT measurement test."
MAX_NEW_TOKENS = 1          # 첫 토큰만 생성 (TTFT 근사)
#NUM_TRIALS = 10             # TTFT 반복 횟수
NUM_TRIALS = 5

# device 설정
DEVICE = "cpu"              # 서버 접속 후 cuda로 변경 가능
#DTYPE = "float16"           # Jetson에서는 fp16 적합, Mac은 bfloat16/float32 fallback
DTYPE = "float32"
