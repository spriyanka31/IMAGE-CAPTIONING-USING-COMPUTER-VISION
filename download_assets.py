# Saikiran Talluri 1002052828 Sai Priyanka Sanku 1002068022 Swarna Ravula 1002033344

from model import Img_Cap_Model
from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor

class Download_Assets:
    def __init__(self, model_path = "assets/model", tokenizer_path = "assets/tokenizer",
                  feature_extractor_path = "assets/feature_extractor"):
        img_cap_model = Img_Cap_Model()
        tokenizer_model = Tokenizer()
        feat_extractor_model = FeatureExtractor()

        img_cap_model.model.save_pretrained(model_path)
        tokenizer_model.tokenizer.save_pretrained(tokenizer_path)
        feat_extractor_model.feature_extractor.save_pretrained(feature_extractor_path)

if __name__ == "__main__":
    Download_Assets()