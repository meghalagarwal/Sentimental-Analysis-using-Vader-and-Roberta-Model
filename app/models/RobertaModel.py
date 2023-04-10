'''Import Libraries'''
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm

class RobertaModel():
    def __init__(self) -> None:
        self.PRETRAINED_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.PRETRAINED_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.PRETRAINED_MODEL)
    
    def polarity_scoring(self, dataframe: object) -> dict:
        result = {}
        for index, rows in tqdm(dataframe.iterrows(), total=len(dataframe)):
            try:
                text = rows['reviewText']
                encoded_text = self.tokenizer(text, return_tensors='pt')
                output = self.model(**encoded_text)
                scores = output[0][0].detach().numpy()
                scores = softmax(scores)
                result[index] = {
                    'roberta_negative': scores[0],
                    'roberta_neutral': scores[1],
                    'roberta_positive': scores[2]
                }
                
            except RuntimeError:
                print(f'Broke for Index: {index}')

        return result