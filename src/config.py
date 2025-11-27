# src/config.py

BACKEND = "vllm"
# 사용할 모델
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"  
MODEL_SIZE_BILLION = 3.2

# TTFT 실험용 프롬프트
TTFT_PROMPT = "Hello, this is a TTFT measurement test."
MAX_NEW_TOKENS = 1
NUM_TRIALS = 10             # TTFT 반복 횟수
MAX_MODEL_LEN = 4096        # vLLM에 전달하는 최대 시퀀스 길이

# ------------------------
# 디바이스 / 백엔드 설정
# ------------------------
DEVICE = "cpu"  # vLLM 쪽에 넘기는 용도
DTYPE = "float32"

# CSV에 기록할 “사람이 읽는 용도의 이름”
DEVICE_NAME = "EWHA-ACPL11_M3_Ultra_CPU"
BACKEND = "vllm_cpu"  # vLLM + CPU backend

# 양자화 안 썼으면 "none"
QUANTIZATION = "none"