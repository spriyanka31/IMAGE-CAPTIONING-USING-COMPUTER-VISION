# Saikiran Talluri 1002052828 Sai Priyanka Sanku 1002068022 Swarna Ravula 1002033344

from transformers import AutoTokenizer

class Tokenizer:
    def __init__(self, model_path = "nlpconnect/vit-gpt2-image-captioning"):
        self.model_path = model_path
        self.load_tokenizer()
    
    def load_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)