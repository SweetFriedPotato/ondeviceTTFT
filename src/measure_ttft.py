# src/measure_ttft.py

import time
import csv
import os
from datetime import datetime

from .config import (
    TTFT_PROMPT,
    MAX_NEW_TOKENS,
    NUM_TRIALS,
    MODEL_ID,
    DEVICE,
    DTYPE
)
from .llm_runner import load_llm

RESULT_DIR = "experiments/results"


def measure_ttft_single(llm, prompt):
    """
    vLLM non-stream 기반 TTFT 측정: 
    max_new_tokens=1 생성에 걸린 시간을 첫 토큰 시간으로 근사.
    """
    from vllm import SamplingParams

    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        top_p=1.0
    )

    start = time.time()
    _ = llm.generate(prompt, params)
    end = time.time()

    return end - start


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"[INFO] Loading model: {MODEL_ID}")
    load_s = time.time()
    llm = load_llm()
    load_e = time.time()

    load_time = load_e - load_s
    print(f"[INFO] Model loaded in {load_time:.3f}s")

    ttft_records = []

    for i in range(NUM_TRIALS):
        t = measure_ttft_single(llm, TTFT_PROMPT)
        print(f"Trial {i+1}/{NUM_TRIALS} TTFT: {t:.4f}s")
        ttft_records.append(t)

    avg_ttft = sum(ttft_records) / len(ttft_records)
    print(f"[RESULT] Average TTFT = {avg_ttft:.4f}s")

    # CSV 저장
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outpath = os.path.join(RESULT_DIR, f"ttft_{timestamp}.csv")

    with open(outpath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial_index", "ttft_seconds"])
        for idx, v in enumerate(ttft_records):
            w.writerow([idx, v])

    print(f"[INFO] Results saved to {outpath}")


if __name__ == "__main__":
    main()
