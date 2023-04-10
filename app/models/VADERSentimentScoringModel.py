'''Import Libraries'''
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
from numba import jit, cuda

class VaderModel():
    def __init__(self) -> None:        
        self.sia = SentimentIntensityAnalyzer()
    
    @jit(target_backend='cuda')
    def polarity_scoring(self, dataframe: object) -> dict:
        result = {}
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            text = row['Processed_data']
            result[index] = self.sia.polarity_scores(text)
        return result