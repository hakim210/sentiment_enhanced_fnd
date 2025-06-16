import os
import json
from tqdm.auto import tqdm

def load_fakenewsnet_dataset(base_path: str, subset: str):
    """
    Args:
        base_path (str): FakeNewsNet 'dataset' 폴더가 있는 상위 경로.
        subset (str): 'PolitiFact' 또는 'GossipCop'

    Returns:
        list of dict: 각 뉴스에 대해 {'text': 뉴스본문, 'label': 0(진짜) or 1(가짜)} 형식의 리스트.
    """
    all_data = []
    subset_path = os.path.join(base_path, 'dataset', subset)
    
    if not os.path.exists(subset_path):
        raise FileNotFoundError(
            f"경로를 찾을 수 없습니다: '{subset_path}'\n"
            f"'{base_path}' 안에 'dataset' 폴더가 있는지, 그 안에 '{subset}' 폴더가 있는지 확인해주세요."
        )

    print(f"'{subset}' 데이터셋 로딩을 시작합니다...")
    for label_name, label_value in [('fake', 1), ('real', 0)]:
        label_dir = os.path.join(subset_path, label_name)
        
        if not os.path.exists(label_dir):
            print(f"경고: '{label_dir}' 폴더가 존재하지 않아 건너뜁니다.")
            continue

        news_ids = os.listdir(label_dir)
        for news_id in tqdm(news_ids, desc=f"Loading {label_name} from {subset}"):
            news_folder = os.path.join(label_dir, news_id)
            news_json_file = os.path.join(news_folder, 'news article.json')

            if os.path.isfile(news_json_file):
                try:
                    with open(news_json_file, 'r', encoding='utf-8') as f:
                        news_content = json.load(f)
                    
                    text = news_content.get('text', '')
                    if text and text.strip():
                        all_data.append({'text': text, 'label': label_value})
                except (json.JSONDecodeError, UnicodeDecodeError):
                    print(f"경고: 파일 로딩/파싱 오류 발생, 건너뜁니다 - {news_json_file}")
                    
    print(f"'{subset}' 데이터셋 로딩 완료. 총 {len(all_data)}개의 뉴스를 불러왔습니다.")
    return all_data

