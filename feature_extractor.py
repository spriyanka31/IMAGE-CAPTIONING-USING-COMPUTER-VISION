# Saikiran Talluri 1002052828 Sai Priyanka Sanku 1002068022 Swarna Ravula 1002033344

from transformers import ViTFeatureExtractor


class FeatureExtractor:
    def __init__(self, model_path = "nlpconnect/vit-gpt2-image-captioning"):
        self.model_path = model_path
        self.load_feature_extractor()
    
    def load_feature_extractor(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_path)