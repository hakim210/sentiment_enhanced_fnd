import torch
from transformers import RobertaTokenizer, RobertaModel

print("Loading RoBERTa model for text embedding... (This may take a moment)")
# Apple Silicon GPU (MPS)가 사용 가능하면 사용하고, 아니면 CPU를 사용
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base').to(device)
print(f"RoBERTa model loaded on device: {device}")

def get_text_embedding(text: str, max_length=128):

    if not isinstance(text, str) or not text.strip():
        return torch.zeros(1, max_length, model.config.hidden_size)

    inputs = tokenizer(
        text, 
        return_tensors='pt', 
        max_length=max_length, 
        padding='max_length', 
        truncation=True
    ).to(device)

    with torch.no_grad(): 
        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state.to('cpu') # CPU로 다시 이동
    
    return last_hidden_states

