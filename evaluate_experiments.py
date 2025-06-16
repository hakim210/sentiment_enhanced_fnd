import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import argparse

from sentiment_enhanced_fnd import (
    config,
    initialize_analyzer, get_sentiment_scores,
    calculate_emotion_weight,
    SentimentGuidedAttention,
    FakeNewsClassifier
)

try:
    from main import setup_model_and_analyzer
except ImportError:
    def setup_model_and_analyzer():
        from huggingface_hub import hf_hub_download
        print("--- 모델 설정 및 초기화 시작 (대체 로직) ---")
        try:
            model_path = hf_hub_download(repo_id=config.REPO_ID, filename=config.MODEL_FILENAME, local_dir=config.LOCAL_MODEL_DIR)
            initialize_analyzer(model_path)
            return True
        except Exception as e:
            print(f"모델 설정 중 오류 발생: {e}")
            return False

def preprocess_for_training(df):
    processed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing training data"):
        text_content = str(row['text']) if pd.notna(row['text']) else ""
        sentiment = get_sentiment_scores(text_content)
        emotion_weight = calculate_emotion_weight(sentiment['polarity'], sentiment['subjectivity'])
        processed.append({
            # 실제 임베딩 대신 더미 텐서를 사용합니다.
            'embedding': torch.randn(1, 128, config.EMBED_DIM),
            'emotion_weight': emotion_weight,
            'label': torch.tensor([row['label']])
        })
    return processed

def train_model_few_shot(train_data, class_weights=None):
    attention_module = SentimentGuidedAttention(embed_dim=config.EMBED_DIM, num_heads=config.NUM_HEADS)
    classifier = FakeNewsClassifier()
    params = list(attention_module.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=3e-5)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    
    attention_module.train()
    classifier.train()
    
    print("Few-shot training started...")
    for epoch in range(20): # 에포크 수는 필요에 따라 조절 가능
        for data in train_data:
            optimizer.zero_grad()
            final_embedding = attention_module(data['embedding'], data['emotion_weight'])
            logits = classifier(final_embedding)
            loss = criterion(logits, data['label'])
            loss.backward()
            optimizer.step()
    print("Few-shot training finished.")
    return {'attention': attention_module, 'classifier': classifier}

def evaluate_model_on_dataset(models, test_df, use_emotion=True, use_beta_weighting=True):
    results = []
    attention_module = models['attention']
    classifier = models['classifier']
    attention_module.eval()
    classifier.eval()
    
    desc_text = f"Evaluating (Emotion: {use_emotion}, Beta Weighting: {use_beta_weighting})"
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc_text):
        text_content = str(row['text']) if pd.notna(row['text']) else ""
        
        if use_emotion:
            beta_to_use = config.BETA if use_beta_weighting else 1.0
            sentiment = get_sentiment_scores(text_content)
            polarity = sentiment.get('polarity', 0.0)
            subjectivity = sentiment.get('subjectivity', 0.5)
            sentiment_score = calculate_emotion_weight(polarity, subjectivity, beta_override=beta_to_use)
        else:
            polarity, subjectivity, sentiment_score = 0.0, 0.0, 1.0
            
        dummy_embedding = torch.randn(1, 128, config.EMBED_DIM)
        with torch.no_grad():
            final_embedding = attention_module(dummy_embedding, sentiment_score)
            logits = classifier(final_embedding)
            probabilities = torch.softmax(logits, dim=1)
            predict_label = torch.argmax(probabilities, dim=1).item()
            
        results.append({
            'id': row['id'], 'polarity': polarity, 'subjectivity': subjectivity,
            'sentiment_score': sentiment_score, 'predict': predict_label,
            'label': row['label']
        })
    return pd.DataFrame(results)

def run_ablation_study(dataset_name, n_shot):
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    RESULTS_DIR = os.path.join(BASE_DIR, "ablation_results")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    filepath = os.path.join(DATASET_DIR, f"{dataset_name}_dataset.csv")

    try:
        df = pd.read_csv(filepath)
        df.dropna(subset=['text'], inplace=True)
        df['text'] = df['text'].astype(str)
    except FileNotFoundError:
        print(f"오류: '{filepath}' 파일을 찾을 수 없습니다. 프로그램을 종료합니다.")
        return

    print(f"\n{'='*20} {dataset_name.upper()} 데이터셋으로 {n_shot}-shot Ablation Study 시작 {'='*20}")
    train_df = df.groupby('label').apply(lambda x: x.sample(n=n_shot, random_state=42)).reset_index(drop=True)
    test_df = df.drop(train_df.index)
    
    print(f"총 {len(df)}개 데이터 -> 학습용: {len(train_df)}개, 테스트용: {len(test_df)}개")
    train_data_processed = preprocess_for_training(train_df)

    # 기본 모델 학습
    print("\n--- [공통] 기본 모델 학습 시작 ---")
    base_models = train_model_few_shot(train_data_processed)
    print("--- [공통] 기본 모델 학습 완료 ---")

    ablation_results = []

    # 실험 1: 베이스라인 (모든 기능 사용)
    print("\n--- [실험 1] 베이스라인 성능 측정 (전체 기능 사용) ---")
    results_baseline_df = evaluate_model_on_dataset(base_models, test_df, use_emotion=True, use_beta_weighting=True)
    f1_baseline = f1_score(results_baseline_df['label'], results_baseline_df['predict'], zero_division=0)
    ablation_results.append({'Experiment': '1. Baseline (All Features)', 'F1-Score': f1_baseline})
    print(f"✅ [베이스라인] F1-Score: {f1_baseline:.4f}")

    # 실험 2: 감성 정보 제거 (Ablation)
    print("\n--- [실험 2] 감성 정보 제거 후 성능 측정 (Ablation) ---")
    results_no_emotion_df = evaluate_model_on_dataset(base_models, test_df, use_emotion=False)
    f1_no_emotion = f1_score(results_no_emotion_df['label'], results_no_emotion_df['predict'], zero_division=0)
    ablation_results.append({'Experiment': '2. Ablation (w/o Emotion)', 'F1-Score': f1_no_emotion})
    print(f"✅ [감성 정보 제거] F1-Score: {f1_no_emotion:.4f}")

    # 실험 3: 부정 감성 가중치(beta) 제거 (Ablation)
    print("\n--- [실험 3] 부정 감성 가중치 제거 후 성능 측정 (Ablation) ---")
    results_no_beta_df = evaluate_model_on_dataset(base_models, test_df, use_emotion=True, use_beta_weighting=False)
    f1_no_beta = f1_score(results_no_beta_df['label'], results_no_beta_df['predict'], zero_division=0)
    ablation_results.append({'Experiment': '3. Ablation (w/o Beta Weight)', 'F1-Score': f1_no_beta})
    print(f"✅ [부정 감성 가중치 제거] F1-Score: {f1_no_beta:.4f}")

    # (선택) 실험 4: 클래스 불균형 해소를 위한 가중치 적용
    print("\n--- [실험 4] 클래스 가중치 적용 후 성능 측정 (Ablation) ---")
    class_counts = df['label'].value_counts()
    if 0 in class_counts and 1 in class_counts and class_counts[0] > 0 and class_counts[1] > 0:
        # 클래스 빈도의 역수를 가중치로 사용
        weights = torch.tensor([1 / class_counts[0], 1 / class_counts[1]], dtype=torch.float32)
        normalized_weights = weights / weights.sum()
        print(f"클래스 가중치 적용: {normalized_weights.tolist()}")
        
        weighted_models = train_model_few_shot(train_data_processed, class_weights=normalized_weights)
        results_weighted_df = evaluate_model_on_dataset(weighted_models, test_df, use_emotion=True, use_beta_weighting=True)
        f1_weighted = f1_score(results_weighted_df['label'], results_weighted_df['predict'], zero_division=0)
        ablation_results.append({'Experiment': '4. Improvement (with Class Weight)', 'F1-Score': f1_weighted})
        print(f"✅ [클래스 가중치 적용] F1-Score: {f1_weighted:.4f}")
    else:
        print("클래스 가중치를 계산할 수 없어 실험 4를 건너뜁니다.")


    summary_df = pd.DataFrame(ablation_results)
    summary_filename = os.path.join(RESULTS_DIR, f"{dataset_name}_{n_shot}shot_ablation_summary.csv")
    summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')

    print(f"\n\n{'='*25}\n✅ Ablation Study 요약 결과가 '{summary_filename}' 파일에 저장되었습니다.\n{'='*25}")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Ablation Study on a Fake News Detection Model.")
    parser.add_argument('--dataset', type=str, default='politifact', choices=['politifact', 'gossipcop'], help='Dataset to use for the study.')
    parser.add_argument('--shot', type=int, default=4, help='Number of shots for few-shot learning.')
    args = parser.parse_args()

    if not setup_model_and_analyzer():
        print("LLM 초기화 실패. 프로그램을 종료합니다.")
        exit()
    
    run_ablation_study(dataset_name=args.dataset, n_shot=args.shot)
