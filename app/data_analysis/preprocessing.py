'''Import Libraries'''
import re
from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import WordNetLemmatizer

class PreProcessing():
    def __init__(self) -> None:
        self.lemma = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
        for wrd in ['don', 'nor', 'or', 'no', 'not', 't', 'against', 'than']:
            self.stop_words.remove(wrd)

    def regular_expression_removal(self, text: str) -> str:
        statement = re.sub('[^a-zA-Z]',' ', text)
        return statement
    
    def lower_case_conversion(self, text: str) -> str:
        statement = text.lower()
        return statement
    
    def stopwords_removal(self, text:str) -> list:
        statement = [word for word in text.split(' ') if word not in set(self.stop_words) and word not in punctuation and word != ' ']
        return statement
    
    def word_lemmatisation(self, text: str) -> list:
        statement = [self.lemma.lemmatize(word) for word in text]
        return statement

    def sentance_processing(self, text: str) -> str:
        new_text = self.regular_expression_removal(text=text)
        new_text = self.lower_case_conversion(text=new_text)
        new_text = self.stopwords_removal(text=new_text)
        new_text = self.word_lemmatisation(text=new_text)
        return ' '.join(str(x) for x in new_text)