# Saikiran Talluri 1002052828 Sai Priyanka Sanku 1002068022 Swarna Ravula 1002033344

from PIL import Image
import torch
from model import Img_Cap_Model
from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor

class Predict:
    def __init__(self):
        self.model = Img_Cap_Model("assets/model").model
        self.tokenizer = Tokenizer("assets/tokenizer").tokenizer
        self.feature_extractor = FeatureExtractor("assets/feature_extractor").feature_extractor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.max_cap_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_cap_length, "num_beams": self.num_beams}

    def predict(self, image):

        if(image.mode != "RGB"):
            image = image.convert(mode = "RGB")
        
        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

        pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return pred