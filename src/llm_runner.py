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
    """
    import os
    import platform
    from .config import MODEL_ID, DTYPE

    IS_APPLE_SILICON = (
        platform.system() == "Darwin" and platform.machine() == "arm64"
    )

    if IS_APPLE_SILICON:
        os.environ["VLLM_USE_V1"] = "0"

    print(f"[INFO] Using vLLM backend on server")
    print(f"[INFO] Model: {MODEL_ID}, dtype: {DTYPE}")

    llm = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        trust_remote_code=True,
        enforce_eager=True,   # torch.compile 끄기
        max_model_len=4096,   # 우리가 쓸 최대 context 길이 (실험용으로 4k면 충분)
        max_num_batched_tokens=4096,  # 위 값과 동일하게 맞춰서 에러 방지
        max_num_seqs=1,       # TTFT 실험은 1개 시퀀스씩만 처리
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
