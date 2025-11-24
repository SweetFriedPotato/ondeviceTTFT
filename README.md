# 온디바이스 LLM — TTFT(Time-To-First-Token) 측정 프로젝트

## 1. 프로젝트 개요
이 프로젝트는 EWHA-ACPL11 Mac Studio(M3 Ultra)에서  
**LLM 모델의 첫 번째 토큰 출력 시간(TTFT)**을 측정하는 실험 환경을 구축하는 것을 목표로 합니다.

vLLM 기반으로 LLaMA 계열 모델을 로딩하고,  
여러 번 반복 실험을 통해 TTFT를 정량화하며  
향후 Jetson/모바일 등 다양한 디바이스와 비교 가능한 구조로 발전시키기 위한 1차 버전입니다.

---

## 2. 디렉토리 구조

```
ttft-project/
├── src/
│     ├── config.py          # 실험 설정값 관리
│     ├── llm_runner.py      # vLLM 기반 모델 로더 및 단일 inference 실행
│     └── measure_ttft.py    # TTFT 측정 메인 스크립트
├── experiments/
│     └── results/           # TTFT 측정 결과(CSV) 자동 저장
├── requirements.txt
├── run.sh
└── README.md
```

---

## 3. 환경 설정

### (1) 가상환경 생성
```
python3.11 -m venv .venv311
```

### (2) 가상환경 활성화  
Mac/Linux:
```
source .venv311/bin/activate
```

Windows:
```
..venv\Scripts\activate
```

### (3) 패키지 설치
```
pip install --upgrade pip
pip install -r requirements.txt
```

### requirements.txt
```
vllm
torch
numpy
pandas
transformers
accelerate
```

---

## 4. 실행 방법

### 방법 1 — 스크립트 실행
```
bash run.sh
```

### 방법 2 — 파이썬 직접 실행
```
python -m src.measure_ttft
```

실행 시 자동으로:
1. 모델 로딩  
2. TTFT N회 측정  
3. 결과 CSV 저장  
이 순서로 진행됩니다.

결과는 다음 위치에 생성됩니다:

```
experiments/results/ttft_YYYYMMDD-HHMMSS.csv
```

---

## 5. 코드 설계 이유

### ✔ 모듈화 구조
- `config.py` 에서 모델명, 디바이스, dtype, 반복 횟수 등 모든 실험 파라미터를 통일 관리  
- `llm_runner.py` 는 모델 로딩·추론만 담당  
- `measure_ttft.py` 는 TTFT 측정 절차만 담당  

→ 디바이스가 바뀌어도 설정만 변경하면 동일 코드로 실험 가능합니다.

### ✔ 재현성 강화
- 모든 TTFT 측정 결과를 CSV로 자동 저장  
- 이후 모델/디바이스 비교 시 재현성 높은 데이터 확보 가능

### ✔ 확장성 고려
- 스트리밍 기반 실제 "첫 토큰 도착 시점" 측정 기능 추가 예정  
- CPU/GPU/NPU 전환, 모델 크기 변경 등 쉽게 확장 가능  
