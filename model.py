# Saikiran Talluri 1002052828 Sai Priyanka Sanku 1002068022 Swarna Ravula 1002033344

from transformers import VisionEncoderDecoderModel

class Img_Cap_Model:
    def __init__(self, model_path = "nlpconnect/vit-gpt2-image-captioning"):
        self.model_path = model_path
        self.load_model()
    
    def load_model(self):
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_path)

