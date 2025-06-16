import torch
from huggingface_hub import hf_hub_download

from sentiment_enhanced_fnd import (
    config,
    calculate_emotion_weight,
    SentimentGuidedAttention,
    FakeNewsClassifier  
)
from sentiment_enhanced_fnd.sentiment_analyzer import initialize_analyzer, get_sentiment_scores

def setup_model_and_analyzer():
    print("--- 모델 설정 및 초기화 시작 ---")
    try:
        model_path = hf_hub_download(
            repo_id=config.REPO_ID,
            filename=config.MODEL_FILENAME,
            local_dir=config.LOCAL_MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"모델 경로: {model_path}")
        initialize_analyzer(model_path)
    except Exception as e:
        print(f"모델 설정 중 심각한 오류 발생: {e}")
        return False
    print("--- 모델 설정 및 초기화 완료 ---\n")
    return True

def main():
    if not setup_model_and_analyzer():
        print("프로그램을 종료합니다.")
        return

    news_text = (
        "The new policy, initially met with optimism, is now facing severe backlash from "
        "small business owners who claim it will lead to widespread bankruptcy and job losses. "
        "Experts are expressing deep concern."
    )
    
    print("--- 1. LLM을 사용한 감성 분석 ---")
    sentiment_data = get_sentiment_scores(news_text)
    polarity = sentiment_data['polarity']
    subjectivity = sentiment_data['subjectivity']
    print(f"분석된 텍스트: \"{news_text[:80]}...\"")
    print(f"결과 -> Polarity: {polarity:.4f}, Subjectivity: {subjectivity:.4f}")

    print("\n--- 2. 부정 감정 강화 가중치 계산 ---")
    emotion_weight = calculate_emotion_weight(polarity, subjectivity)
    print(f"계산된 감성 가중치: {emotion_weight:.4f} (α={config.ALPHA}, β={config.BETA})")

    print("\n--- 3. Attention 기반 감성 통합 ---")
    dummy_embedding = torch.randn(1, 128, config.EMBED_DIM)
    attention_module = SentimentGuidedAttention(embed_dim=config.EMBED_DIM, num_heads=config.NUM_HEADS)
    final_embedding = attention_module(dummy_embedding, emotion_weight)
    print("감성 정보가 통합된 최종 텍스트 임베딩이 생성되었습니다.")
    print(f"최종 임베딩 Shape: {final_embedding.shape}")

    print("\n--- 4. 최종 뉴스 진위 분류 ---")
    classifier = FakeNewsClassifier()
    logits = classifier(final_embedding)
    probabilities = torch.softmax(logits, dim=1)
    predicted_index = torch.argmax(probabilities, dim=1).item()
    predicted_label = config.CLASS_LABELS[predicted_index]
    
    print(f"분류 결과 (로짓): {logits.detach().numpy()}")
    print(f"분류 결과 (확률): {probabilities.detach().numpy()}")
    print("-" * 30)
    print(f"최종 예측 결과: 이 뉴스는 '{predicted_label}' 뉴스일 확률이 높습니다.")
    print("-" * 30)

if __name__ == "__main__":
    main()

