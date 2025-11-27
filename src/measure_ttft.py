# src/measure_ttft.py

import os
import time
import csv
from datetime import datetime

from .config import (
    TTFT_PROMPT,
    MAX_NEW_TOKENS,
    NUM_TRIALS,
    MODEL_ID,
    MODEL_SIZE_BILLION,
    DEVICE_NAME,
    BACKEND,
    DTYPE,
    MAX_MODEL_LEN,
    QUANTIZATION,
)
from .llm_runner import create_llm

RESULT_DIR = "experiments/results"


def measure_ttft_single(llm, prompt: str, max_new_tokens: int = 1) -> float:
    """
    vLLM non-stream 모드에서 TTFT 측정:
    max_new_tokens=1 생성에 걸린 전체 시간을 첫 토큰 시간으로 근사.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    start = time.time()
    _ = llm.generate(prompt, params)
    end = time.time()

    return end - start


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"[INFO] Loading model: {MODEL_ID}")
    load_s = time.time()
    llm = create_llm()
    load_e = time.time()
    load_time = load_e - load_s
    print(f"[INFO] Model loaded in {load_time:.3f} seconds")

    # ---- 프롬프트 토큰 길이 계산 (한 번만) ----
    try:
        tokenizer = llm.get_tokenizer()
        prompt_length_tokens = len(tokenizer.encode(TTFT_PROMPT))
    except Exception:
        # 혹시라도 토크나이저 접근이 안 되면 fallback
        prompt_length_tokens = -1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULT_DIR, f"ttft_{timestamp}.csv")

    records = []
    ttft_list = []

    for i in range(NUM_TRIALS):
        print(f"[INFO] Trial {i + 1}/{NUM_TRIALS}")
        ttft = measure_ttft_single(llm, TTFT_PROMPT, max_new_tokens=MAX_NEW_TOKENS)
        ttft_list.append(ttft)

        is_warmup = 1 if i == 0 else 0

        print(f"  -> TTFT: {ttft:.4f} s (is_warmup={is_warmup})")

        record = {
            "timestamp": timestamp,
            "device_name": DEVICE_NAME,
            "backend": BACKEND,
            "model_id": MODEL_ID,
            "model_size_billion": MODEL_SIZE_BILLION,
            "dtype": DTYPE,
            "max_model_len": MAX_MODEL_LEN,
            "quantization": QUANTIZATION,
            "num_trials": NUM_TRIALS,
            "trial_index": i,
            "is_warmup": is_warmup,
            "prompt_length_tokens": prompt_length_tokens,
            "ttft_seconds": ttft,
            "model_load_time_seconds": load_time,
        }
        records.append(record)

    avg_ttft = sum(ttft_list) / len(ttft_list)
    print(f"[RESULT] Average TTFT: {avg_ttft:.4f} s")

    # ---- CSV 저장 ----
    fieldnames = [
        "timestamp",
        "device_name",
        "backend",
        "model_id",
        "model_size_billion",
        "dtype",
        "max_model_len",
        "quantization",
        "num_trials",
        "trial_index",
        "is_warmup",
        "prompt_length_tokens",
        "ttft_seconds",
        "model_load_time_seconds",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    print(f"[INFO] Saved results to {csv_path}")


if __name__ == "__main__":
    main()
