# Domain-Specific Sentiment Weighting for Political Fake News Detection

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 가짜 뉴스 탐지(Fake News Detection)의 정확도를 높이기 위해 **뉴스 기사에 내재된 감성 정보(Emotional Elements)**를 적극적으로 활용하는 모델을 개발하고 평가합니다[5][8][12][21]. [AMPLE: Emotion-Aware Multimodal Fusion Prompt Learning for Fake News Detection](https://arxiv.org/abs/2410.15591)[1] 논문에서 영감을 받아, 텍스트의 감성적 특징을 추출하고 이를 어텐션 메커니즘과 결합하는 아이디어를 구현하고 확장했습니다.

### 주요 목표
이 프로젝트는 다음과 같은 가설을 설정하고 검증하는 것을 목표로 합니다.

1.  **LLM 활용 감성 분석**: 기존 연구는 `TextBlob`이라는 lexicon-based 방식으로 감성점수를 추출했으나, 본 연구에서는 **대규모 언어 모델(LLM)을 활용**하여 더 깊고 정확한 감성 분석을 수행하고, 이를 통해 모델 성능을 개선할 수 있는 가능성을 탐구했습니다[6][7][22]. (LLM 적용 과정에서 긴 텍스트 처리를 위한 'Chunk & Aggregate' 기법을 활용했습니다.)
2.  **부정 감성 강화**: "가짜 뉴스는 특히 부정적인 감성을 더 강하게 나타난다"는 가설[1, 7]에 기반하여, 감성 점수 계산 시 **부정 감성에 더 높은 가중치(`BETA` 계수)를 부여**하는 새로운 로직을 설계하고 그 효과를 검증합니다.
3.  **커스텀 어텐션 융합**: 추출된 감성 정보를 텍스트의 의미 정보와 효과적으로 융합하기 위해, 감성 가중치를 Self-Attention 스코어에 직접 주입하는 **`SentimentGuidedAttention`** 메커니즘을 설계했습니다.

이 프로젝트는 FakeNewsNet의 PolitiFact와 GossipCop 데이터셋을 사용하여 위 가설들을 검증하기 위한 실험을 수행했습니다.

## 2. 모델 아키텍처 (Model Architecture)

모델은 **(A) 데이터 입력**, **(B) 감성-텍스트 융합 특성 추출**, **(C) 최종 분류**의 3단계 파이프라인으로 구성됩니다. 특히 (B) 단계에서 감성 정보와 텍스트 의미 정보를 융합했습니다.

```
insert image
```

## 3. 파일 및 폴더 구조

프로젝트는 소스 코드, 데이터, 실험 스크립트, 결과물로 구분됩니다.

```
HCI_project/
│
├── sentiment_enhanced_fnd/      # 📦 핵심 기능을 담은 Python 소스 코드 패키지
│   ├── __init__.py              #    - 패키지 초기화 및 모듈 공개
│   ├── config.py                #    - 모델 하이퍼파라미터 및 설정 관리
│   ├── sentiment_analyzer.py    #    - 감성 분석(Polarity, Subjectivity) 로직
│   ├── emotion.py               #    - '부정 감정 강화 가중치' 계산 로직
│   ├── text_embedder.py         #    - RoBERTa 모델을 이용한 텍스트 임베딩 로직 (New)
│   ├── attention.py             #    - 'SentimentGuidedAttention' 모듈 정의
│   └── classifier.py            #    - 최종 진위 판별 분류기 정의
│
├── dataset/                     # 🗂️ 데이터셋 저장 위치
│   ├── politifact_dataset.csv
│   └── gossipcop_dataset.csv
│
├── evaluate_experiments.py      # 🔬 실험 실행 및 결과 생성 메인 스크립트
├── calculate_metrics.py         # 📊 생성된 결과 파일을 분석하고 성능 지표를 계산/시각화하는 스크립트
├── main.py                      # ⚙️ 모델의 기본 파이프라인 및 초기화 로직
├── README.md                    # 📄 (현재 파일) 프로젝트 설명서
└── requirements.txt             # 📜 필요한 라이브러리 목록
```

## 4. 설치 및 실행 방법

### 4.1. 환경 설정

1.  **저장소 복제**
    ```
    git clone <url>
    cd @@@
    ```
2.  **필수 라이브러리 설치**
    ```
    pip install -r requirements.txt
    ```
### 4.2. 데이터 준비

1.  이전 단계에서 생성한 `politifact_dataset.csv`와 `gossipcop_dataset.csv` 파일이 `dataset/` 폴더 안에 위치하고 있는지 확인합니다.

### 4.3. 실험 실행

1.  **모델 평가 및 결과 생성**
     `evaluate_on_datasets.py` 스크립트를 실행하여 PolitiFact 데이터셋과 Gossipcop 데이터셋에 대한 모델의 적용 결과를 CSV 파일로 생성합니다.
    ```
    python3 evaluate_on_datasets.py
    
3.  **성능 지표 분석**
    `evaluate_experiments.py` 스크립트를 실행하여 PolitiFact 데이터셋에 대한 실험을 수행하고 결과 CSV 파일들을 생성합니다.
    ```
    python3 evaluate_experiments.py


## 5. 실험 내용 및 의의

`evaluate_on_datasets.py` 스크립트의 실행결과로 PolitiFact 데이터셋과 Gossipcop 데이터셋의 실험 결과에 대한 분석결과(Confusion Matrix, Accuracy, Precision, Recall, F1-score)는 Evaluation.ipynb 파일에서 확인할 수 있습니다.

본 프로젝트에서는 모델의 성능과 각 구성 요소의 효과를 검증하기 위해 다음과 같은 실험을 추가로 수행했습니다.

### 실험 1: 클래스 불균형 문제 해결

*   **가설**: PolitiFact 데이터셋에서 모델이 '가짜 뉴스'를 잘 탐지하지 못하고 '진짜 뉴스'로 편향되는 현상은 클래스 불균형 때문일 수 있다.
*   **방법**: 소수 클래스인 '가짜 뉴스(label=1)'를 틀렸을 때 더 큰 페널티를 부여하는 **가중 손실 함수(Weighted Cross-Entropy Loss)**를 적용하여 모델을 학습시켰습니다.
*   **의의**: 모델이 단순히 전체 정확도만 높이는 것이 아니라, 더 중요한 소수 클래스(가짜 뉴스)에 대한 탐지 능력(재현율, Recall)을 향상시킬 수 있는지 검증합니다.

### 실험 2: 감성 분석의 효과 검증 (Ablation study 1)

*   **가설**: 우리가 설계한 "감성 분석 → 가중치 계산 → 어텐션 융합" 파이프라인은 가짜 뉴스 탐지 성능에 긍정적인 영향을 미친다.
*   **방법**: AMPLE 논문의 **-EE (Emotional Elements 제거)** 실험과 동일하게[1,3], 아래 두 모델의 성능을 F1-Score로 비교했습니다[5].
    1.  **With Emotion**: 감성 정보를 모두 활용하는 완전한 모델
    2.  **Without Emotion**: 감성 정보를 사용하지 않고(`sentiment_score=1.0`), 오직 텍스트 임베딩만으로 예측하는 모델
*   **의의**: 감성 정보가 실제로 모델 성능에 얼마나 기여하는지를 정량적으로 측정하여, 프로젝트의 핵심 아이디어의 유효성을 입증합니다.

### 실험 3: 부정 감정 강화 로직 검증 (Ablation study 2)

*   **가설**: 우리가 독자적으로 설계한 "부정 감정 강화 로직(`BETA` 계수)"은 가짜 뉴스 탐지에 추가적인 성능 향상을 가져온다.
*   **방법**: 감성 분석을 사용하되, 오직 `BETA` 계수의 효과만을 비교했습니다.
    1.  **With Beta Weighting**: `BETA=1.5`로 설정하여 부정 감성을 강화한 모델
    2.  **Without Beta Weighting**: `BETA=1.0`으로 설정하여 긍정/부정 감성을 동등하게 취급한 모델
*   **의의**: AMPLE 논문을 넘어서는 우리만의 독창적인 기여가 실제로 유효한지 검증하는 가장 중요한 실험입니다.

## 6. 향후 연구 방향 (Future Work)

*   **End-to-End Fine-tuning**: 현재는 RoBERTa를 고정된 특징 추출기로 사용하지만, 전체 모델을 End-to-End로 미세 조정하여 성능을 추가적으로 향상
*   **하이퍼파라미터 최적화**: `ALPHA`, `BETA` 값 등 주요 하이퍼파라미터에 대한 체계적인 탐색(Grid Search 등)을 통해 최적의 조합을 탐구
*   **멀티모달 확장**: AMPLE 논문의 원래 아이디어처럼, 이미지 정보를 CLIP 인코더로 추출하고 이를 텍스트/감성 정보와 융합하여 모델의 성능을 더욱 고도화

## 7. 참고 문헌 (References)
[1] X. Xu, X. Li, T. Wang, and Y. Jiang, "AMPLE: Emotion-Aware Multimodal Fusion Prompt Learning for Fake News Detection," *arXiv preprint arXiv:2410.15591*, 2024.

[2] Cambridge Dictionary, "False news," [Online]. Available: https://dictionary.cambridge.org/dictionary/english/false-news

[3] Ethical Journalism Network, "What is Fake News?," [Online]. Available: https://ethicaljournalismnetwork.org/resources/publications/ejn-report-on-fake-news

[4] PolitiFact, "PolitiFact Statements Dataset," [Online]. Available: https://www.politifact.com/

[5] A. K. Dey, S. Saha, and S. Saha, "Sentiment Analysis for Fake News Detection by Means of Neural Networks," *Journal of Biomedical Informatics*, vol. 109, p. 103514, 2020.

[6] M. Yang, L. Flores, H. Hunma, and B. Trevisan, "Exploring the Impact of Sentiment Analysis on Current Methods of Fake News Detection," *Yale Undergraduate Research Journal*, vol. 3, no. 1, 2022.

[7] A. Kuila and S. Sarkar, "Deciphering Political Entity Sentiment in News with Large Language Models: Zero-Shot and Few-Shot Strategies," in *Proceedings of the 4th Workshop on NLP for Political Text (PoliticalNLP 2024)*, 2024.

[8] S. Roy, D. Paul, and C. Das, "Fake News Detection: Comparative Evaluation of BERT-like Models and Large Language Models with Generative AI-Annotated Data," *arXiv preprint arXiv:2412.14276*, 2024.

[9] S. Bondielli and F. Marcelloni, "Sentiment Analysis for Fake News Detection," *ProQuest*, 2021.

[10] K. Shu, A. Sliva, S. Wang, J. Tang, and H. Liu, "Fake News Detection on Social Media: A Data Mining Perspective," *SIGKDD Explorations*, vol. 19, no. 1, pp. 22–36, 2017.

[11] ISOT Research Lab, "ISOT Fake News Dataset," [Online]. Available: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news/index.php

[12] A. J. Bidgoly, "FNID: Fake News Inference Dataset," *IEEE DataPort*, 2020.

[13] S. V. Balshetwar, R. S. Abilash, and R. D. Jermisha, "Fake news detection in social media based on sentiment analysis using classifier techniques," *Multimedia Tools and Applications*, vol. 82, pp. 18653–18678, 2023.

[14] I. K. Sastrawan, I. P. A. Bayupati, and D. M. S. Arsa, "Detection of fake news using deep learning CNN–RNN based methods," *ICT Express*, vol. 8, no. 3, pp. 354–360, 2022.

[15] A. Saha, T. Khondakar, M. B. Miah, S. A. Ahsan, and D. Nandi, "A Comparative Analysis on Fake News Detection Methods," in *Proceedings of the 2022 International Conference on Computer, Communication, Chemical, Material and Electronic Engineering (IC4ME2)*, 2022, pp. 1–6.

[16] "A Semi-supervised Fake News Detection using Sentiment Encoding," *arXiv preprint arXiv:2407.19332*, 2024.

[17] "Domain Adaptive Fake News Detection via Reinforcement Learning," *arXiv preprint arXiv:2202.08159*, 2024.

[18] "Cross-Domain Fake News Detection based on Dual-Granularity Adversarial Training," in *Proceedings of the 29th International Conference on Computational Linguistics (COLING 2025)*, 2025.

[19] F. Hamborg, "NewsMTSC: Target-dependent sentiment classification in news articles," GitHub, 2021. [Online]. Available: https://github.com/fhamborg/NewsMTSC

[20] H. Hamed, A. G. Al-Khafaji, and S. Al-Sumaidaee, "Fake News Detection Model on Social Media by Leveraging Sentiment Analysis of News Content and Emotion Analysis of Users’ Comments," *Sensors*, vol. 23, 2023.

[21] D. Vilares, A. O. C. E. Alonso, and M. G. de la Fuente, "Sentiment analysis for fake news detection," *Electronics*, vol. 10, 2021.
