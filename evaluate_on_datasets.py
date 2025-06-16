import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm.auto import tqdm
import random
import os
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

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
        print("--- 모델 설정 및 초기화 시작 ---")
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
            'embedding': torch.randn(1, 128, config.EMBED_DIM), 'emotion_weight': emotion_weight,
            'label': torch.tensor([row['label']])
        })
    return processed

def train_model_few_shot(train_data):
    attention_module = SentimentGuidedAttention(embed_dim=config.EMBED_DIM, num_heads=config.NUM_HEADS)
    classifier = FakeNewsClassifier()
    params = list(attention_module.parameters()) + list(classifier.parameters())
    optimizer = optim.AdamW(params, lr=3e-5)
    criterion = nn.CrossEntropyLoss()
    attention_module.train()
    classifier.train()
    print("Few-shot training started...")
    for epoch in range(20):
        for data in train_data:
            optimizer.zero_grad()
            final_embedding = attention_module(data['embedding'], data['emotion_weight'])
            logits = classifier(final_embedding)
            loss = criterion(logits, data['label'])
            loss.backward()
            optimizer.step()
    print("Few-shot training finished.")
    return {'attention': attention_module, 'classifier': classifier}

def evaluate_model_on_dataset(models, test_df):
    results = []
    attention_module = models['attention']
    classifier = models['classifier']
    attention_module.eval()
    classifier.eval()
    
    desc_text = f"Evaluating on {len(test_df)} articles"
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=desc_text):
        text_content = str(row['text']) if pd.notna(row['text']) else ""
        sentiment = get_sentiment_scores(text_content)
        polarity = sentiment.get('polarity', 0.0)
        subjectivity = sentiment.get('subjectivity', 0.5)
        sentiment_score = calculate_emotion_weight(polarity, subjectivity)
        
        dummy_embedding = torch.randn(1, 128, config.EMBED_DIM)
        with torch.no_grad():
            final_embedding = attention_module(dummy_embedding, sentiment_score)
            logits = classifier(final_embedding)
            probabilities = torch.softmax(logits, dim=1)
            predict_label = torch.argmax(probabilities, dim=1).item()
            
        results.append({
            'id': row['id'], 'polarity': polarity, 'subjectivity': subjectivity,
            'sentiment_score': sentiment_score, 'true_percentage': probabilities[0,0].item(),
            'false_percentage': probabilities[0,1].item(), 'predict': predict_label,
            'label': row['label']
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    if not setup_model_and_analyzer():
        print("LLM 초기화 실패. 프로그램을 종료합니다.")
        exit()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
    
    DETAILED_RESULTS_DIR = os.path.join(BASE_DIR, "detailed_shot_results")
    os.makedirs(DETAILED_RESULTS_DIR, exist_ok=True)
    
    shot_sizes = [4, 16, 64]
    all_experiment_results = [] 

    datasets_to_process = {
        "politifact": os.path.join(DATASET_DIR, "politifact_dataset.csv"),
        "gossipcop": os.path.join(DATASET_DIR, "gossipcop_dataset.csv")
    }

    for name, filepath in datasets_to_process.items():
        print(f"\n{'='*25}\n   데이터셋 처리 시작: {name.upper()}\n{'='*25}")
        try:
            df_original = pd.read_csv(filepath)
            df_original.dropna(subset=['text'], inplace=True)
            df_original['text'] = df_original['text'].astype(str)
        except FileNotFoundError:
            print(f"오류: '{filepath}' 파일을 찾을 수 없습니다. 건너뜁니다.")
            continue

        df = df_original.copy()

        if name == "gossipcop":
            fake_news = df[df['label'] == 1]
            real_news = df[df['label'] == 0]
            
            num_samples = 120
            if len(fake_news) < num_samples or len(real_news) < num_samples:
                print(f"경고: GossipCop의 진짜 또는 가짜 뉴스가 {num_samples}개 미만이라 건너뜁니다.")
                continue
            
            fake_sample = fake_news.sample(n=num_samples, random_state=42)
            real_sample = real_news.sample(n=num_samples, random_state=42)
            df = pd.concat([fake_sample, real_sample]).sample(frac=1, random_state=42).reset_index(drop=True)
            print(f"새로운 GossipCop 서브셋 생성 완료 (총 {len(df)}개, 라벨 분포: \n{df['label'].value_counts().to_string()})")

        for shot in shot_sizes:
            print(f"\n--- {name.upper()} 데이터셋, {shot}-shot 실험 시작 ---")
            num_samples_per_class = shot
            if len(df[df['label']==0]) < num_samples_per_class or len(df[df['label']==1]) < num_samples_per_class:
                print(f"경고: {shot}-shot 실험을 위한 데이터가 부족하여 건너뜁니다.")
                continue
            
            train_df = df.groupby('label').apply(lambda x: x.sample(n=num_samples_per_class, random_state=42)).reset_index(drop=True)
            test_df = df.drop(train_df.index)

            print(f"총 {len(df)}개 데이터 -> 학습용: {len(train_df)}개, 테스트용: {len(test_df)}개")
            
            train_data_processed = preprocess_for_training(train_df)
            trained_models = train_model_few_shot(train_data_processed)
            results_df = evaluate_model_on_dataset(trained_models, test_df)
            
            detailed_output_filename = os.path.join(DETAILED_RESULTS_DIR, f"{name}_{shot}shot_detailed_results.csv")
            results_df.to_csv(detailed_output_filename, index=False, encoding='utf-8-sig')
            print(f"💾 상세 예측 결과가 '{detailed_output_filename}' 파일에 저장되었습니다.")

            y_true = results_df['label']
            y_pred = results_df['predict']
            
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            print(f"✅ {shot}-shot 결과: Accuracy: {accuracy:.2%}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            
            all_experiment_results.append({
                'dataset': name,
                'shot_size': shot,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })

    if all_experiment_results:
        final_results_df = pd.DataFrame(all_experiment_results)
        output_filename = os.path.join(BASE_DIR, "few_shot_learning_summary_results.csv")
        final_results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n\n{'='*25}\n✅ 모든 실험의 요약 결과가 '{output_filename}' 파일에 저장되었습니다.\n{'='*25}")
        print("최종 요약 결과:")
        print(final_results_df)

