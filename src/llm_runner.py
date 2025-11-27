# src/llm_runner.py

import os
import platform
from vllm import LLM, SamplingParams

from .config import MODEL_ID, DTYPE

# Apple Silicon인지 체크 (M3 Ultra 포함)
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and platform.machine() == "arm64"
)


def create_llm():
    """
    서버(M3 Ultra)에서 vLLM을 사용해 LLM을 로드하는 함수.
    - Apple Silicon + CPU-only 환경에서 안정적으로 돌리기 위해:
      - VLLM_USE_V1=0 으로 V1 엔진 비활성화
      - enforce_eager=True 로 torch.compile / Inductor 비활성화
    """

    if IS_APPLE_SILICON:
        # V1 엔진 끄기 (ARM + CPU에서 문제 많은 부분)
        os.environ["VLLM_USE_V1"] = "0"

    print(f"[INFO] Using vLLM backend on server")
    print(f"[INFO] Model: {MODEL_ID}, dtype: {DTYPE}")

    llm = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        trust_remote_code=True,
        enforce_eager=True,  # CPU 환경에서 Inductor 끄는 핵심 옵션
    )
    return llm


def generate_once(llm, prompt: str, max_tokens: int) -> str:
    """
    vLLM을 이용해 한 번 생성하는 함수.
    TTFT 측정에서는 max_tokens를 1로 두고,
    첫 토큰이 나오는 전체 지연시간을 재는 용도로 사용.
    """
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    outputs = llm.generate(prompt, sampling_params)
    text = outputs[0].outputs[0].text
    return text
