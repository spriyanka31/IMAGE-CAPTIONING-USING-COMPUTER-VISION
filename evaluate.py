# Saikiran Talluri 1002052828 Sai Priyanka Sanku 1002068022 Swarna Ravula 1002033344

from predict import Predict
from nltk.translate.bleu_score import sentence_bleu

class Evaluate:
    def __init__(self, image, reference):
        self.reference = reference
        self.predict = Predict()
        self.generated_cap = self.predict.predict(image)
        self.bleu_score = self.calculate_bleu()   

    def calculate_bleu(self):
        self.reference = [self.reference.split()]
        self.generated_cap = self.generated_cap.split()
        return sentence_bleu(self.reference, self.generated_cap)
    

