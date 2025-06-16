import os

# --- 모델 설정 (Hugging Face Hub) ---
REPO_ID = "NoelJacob/Meta-Llama-3-8B-Instruct-Q4_K_M-GGUF"
MODEL_FILENAME = "meta-llama-3-8b-instruct.Q4_K_M.gguf"

# 모델을 저장할 로컬 디렉토리
LOCAL_MODEL_DIR = "models"


# --- LLM 로더 설정 ---
N_GPU_LAYERS = -1  # Metal GPU 사용을 위해 -1 또는 1 이상의 값 설정
N_CTX = 2048       # 모델의 컨텍스트 크기


# --- 감성 가중치 하이퍼파라미터 ---
ALPHA = 1.0  # 감성 가중치 영향력 계수
BETA = 1.5   # 부정 감정 강화 계수


# --- Attention 모듈 하이퍼파라미터 ---
EMBED_DIM = 768  # 텍스트 임베딩 차원
NUM_HEADS = 12   # 어텐션 헤드 수


# --- 분류기 설정 ---
NUM_CLASSES = 2  # 클래스 수 (True, False)
CLASS_LABELS = {0: "TRUE", 1: "FALSE"} # 인덱스에 따른 레이블
