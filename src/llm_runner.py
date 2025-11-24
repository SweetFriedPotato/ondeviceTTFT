# src/llm_runner.py

from vllm import LLM, SamplingParams
from .config import MODEL_ID, DTYPE

def load_llm():
    """
    vLLM 기반으로 LLM 로드.
    Mac에서도 CPU fallback으로 동작하도록 설계.
    """
    llm = LLM(
        model=MODEL_ID,
        dtype=DTYPE,
        trust_remote_code=True
    )
    return llm


def generate_once(llm, prompt: str, max_tokens: int = 16):
    """
    LLM 한 번 실행해보기 — 초기 디버깅용
    """
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0
    )

    outputs = llm.generate(prompt, params)
    text = outputs[0].outputs[0].text
    return text
