# src/measure_ttft.py

import time
import csv
import os
from datetime import datetime

from .config import (
    TTFT_PROMPT,
    MAX_NEW_TOKENS_FOR_TTFT,
    NUM_TRIALS,
    MODEL_ID,
)
from .llm_runner import create_llm, generate_once

RESULT_DIR = "experiments/results"


def measure_ttft_once(llm, prompt: str, max_tokens: int) -> float:
    """
    한 번의 요청에 대해
    - generate_once 호출 전후 시간 차이를 재서
    - 첫 번째 토큰이 나올 때까지의 지연시간을 TTFT로 근사.
    """
    start = time.time()
    _ = generate_once(llm, prompt, max_tokens)
    end = time.time()
    return end - start


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print(f"[INFO] Loading model: {MODEL_ID}")
    load_start = time.time()
    llm = create_llm()
    load_end = time.time()
    load_time = load_end - load_start
    print(f"[INFO] Model loaded in {load_time:.3f} seconds")

    ttft_list = []

    for i in range(NUM_TRIALS):
        print(f"[INFO] Trial {i+1}/{NUM_TRIALS}")
        ttft = measure_ttft_once(
            llm,
            TTFT_PROMPT,
            MAX_NEW_TOKENS_FOR_TTFT,
        )
        print(f"  -> TTFT: {ttft:.4f} s")
        ttft_list.append(ttft)

    avg_ttft = sum(ttft_list) / len(ttft_list)
    print(f"[RESULT] Average TTFT: {avg_ttft:.4f} s")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULT_DIR, f"ttft_{timestamp}.csv")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "model_id",
            "num_trials",
            "trial_index",
            "ttft_seconds",
            "model_load_time_seconds",
        ])
        for idx, v in enumerate(ttft_list):
            writer.writerow([
                timestamp,
                MODEL_ID,
                NUM_TRIALS,
                idx,
                v,
                load_time,
            ])

    print(f"[INFO] Saved results to {out_path}")


if __name__ == "__main__":
    main()
