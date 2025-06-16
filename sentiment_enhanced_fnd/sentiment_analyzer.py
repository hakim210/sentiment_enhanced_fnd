import json
import os
from llama_cpp import Llama
from . import config
import math

llm_instance = None
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert sentiment analysis tool. Your task is to analyze the provided news text and output its sentiment properties in a specific JSON format.
1.  **Sentiment Polarity**: Determine the sentiment polarity on a continuous scale from -1.0 (very negative) to 1.0 (very positive). A score of 0.0 represents a neutral sentiment.
2.  **Subjectivity**: Determine the subjectivity on a continuous scale from 0.0 (very objective) to 1.0 (very subjective).
Provide your answer ONLY as a single, clean JSON object with two keys: "polarity" and "subjectivity".<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze the following text:

{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

def initialize_analyzer(model_path: str):
    global llm_instance
    try:
        llm_instance = Llama(
            model_path=model_path,
            n_gpu_layers=config.N_GPU_LAYERS,
            n_ctx=config.N_CTX,
            verbose=False
        )
        print(f"Sentiment Analyzer: '{os.path.basename(model_path)}' 모델이 성공적으로 로드되었습니다.")
    except Exception as e:
        print(f"Sentiment Analyzer: 모델 로드 중 오류 발생 - {e}")
        llm_instance = None

def get_sentiment_scores(text: str) -> dict:

    if not llm_instance:
        print("오류: 분석기가 초기화되지 않았습니다.")
        return {"polarity": 0.0, "subjectivity": 0.5}

    if not isinstance(text, str) or not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.5}

    max_chunk_chars = (config.N_CTX - 200) * 3
    overlap_chars = int(max_chunk_chars * 0.1) # 10% 겹치게 하여 문맥 유지
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chunk_chars
        chunks.append(text[start:end])
        start += max_chunk_chars - overlap_chars
    
    if not chunks:
        return {"polarity": 0.0, "subjectivity": 0.5}

    polarities = []
    subjectivities = []

    for chunk in chunks:
        if not chunk.strip():
            continue
            
        prompt = PROMPT_TEMPLATE.format(text=chunk)
        
        try:
            output = llm_instance(prompt, max_tokens=100, temperature=0.0, stop=["<|eot_id|>"])
            response_text = output['choices'][0]['text'].strip()
            
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            sentiment_data = json.loads(json_str)
            
            polarities.append(float(sentiment_data.get('polarity', 0.0)))
            subjectivities.append(float(sentiment_data.get('subjectivity', 0.5)))
        except Exception:
            continue

    if not polarities: 
        return {"polarity": 0.0, "subjectivity": 0.5}

    # Polarity: 전체적인 경향을 보기 위해 평균값을 사용
    avg_polarity = sum(polarities) / len(polarities)
    
    # Subjectivity: 기사의 어느 한 부분이라도 주관적이면 전체가 주관적일 수 있으므로, 최댓값을 사용
    max_subjectivity = max(subjectivities)
    
    return {"polarity": avg_polarity, "subjectivity": max_subjectivity}
