'''Import Libraries'''
import os
import pandas as pd
import ndjson
import requests
import nltk
import matplotlib.pyplot as plt
from data_analysis.analysis import DataAnalysis
from data_analysis.preprocessing import PreProcessing
from models.VADERSentimentScoringModel import VaderModel
from models.RobertaModel import RobertaModel
from os.path import exists
from tqdm import tqdm
tqdm.pandas()
import wget
import gzip

'''Objects of Class and Models'''
d_analysis = DataAnalysis()
nltk.download(['stopwords','wordnet', 'vader_lexicon'])
data_preprocessing = PreProcessing()
vader_model = VaderModel()
roberta_model = RobertaModel()

'''Application start here'''
if __name__ == "__main__":
    '''Checking if the data is already converted to csv file'''
    if not exists('..\data\\videogamesreview.csv'):
        # wget.download('https://jmcauley.ucsd.edu/data/amazon_v2/categoryFiles/Video_Games.json.gz')

        with gzip.open('D:\Projects\Personal Projects\Sentiment-Analysis-Using-Vader-And-Roberta-Models\data\Video_Games.json.gz', 'rb') as f:
            video_games_data = pd.DataFrame(ndjson.load(f))

        video_games_data.to_csv('D:/Projects/Personal Projects/Sentiment-Analysis-Using-Vader-And-Roberta-Models/data/videogamesreview.csv', index=False)
        os.remove('D:\Projects\Personal Projects\Sentiment-Analysis-Using-Vader-And-Roberta-Models\data\Video_Games.json.gz')
    else:
        video_games_data = pd.read_csv('..\data\\videogamesreview.csv')

    '''Feature Selection based on requirements'''
    review_data = video_games_data[['overall', 'reviewerName', 'reviewText']]

    '''Spliting the data into training and testing'''
    train_review_data = review_data[:100] #.sample(frac=0.25)

    '''Understanding of Data by measure of Central tendency and Dispersion'''
    d_analysis.understanding_data(train_review_data)

    '''Graph representation of distribution of over all rating'''
    d_analysis.overall_rating_distribution(train_review_data[['overall']])

    '''Preprocessing of data for machine understandable'''
    train_review_data['Processed_data'] = train_review_data['reviewText'].progress_apply(lambda text: data_preprocessing.sentance_processing(text=str(text)))

    ''''''
    vader_temp_data = pd.DataFrame(vader_model.polarity_scoring(train_review_data)).T.rename(columns={'neg': 'vader_negative', 'neu': 'vader_neutral', 'pos': 'vader_positive', 'compound': 'vader_compound'})
    train_review_data = pd.merge(train_review_data, vader_temp_data, left_index=True, right_index=True)

    roberta_temp_data = pd.DataFrame(roberta_model.polarity_scoring(train_review_data)).T
    train_review_data = pd.merge(train_review_data, roberta_temp_data, left_index=True, right_index=True)

    fx, ax = plt.subplots(2, 3, figsize=(20, 5))
    d_analysis.validation_scoring(title='Vader\'s Positive Score by Video Games Star Review', dataframe=train_review_data, x_axis='overall', y_axis='vader_positive', ax=ax[0][0])
    d_analysis.validation_scoring(title='Vader\'s Negative Score by Video Games Star Review', dataframe=train_review_data, x_axis='overall', y_axis='vader_negative', ax=ax[0][1])
    d_analysis.validation_scoring(title='Vader\'s Compound Score by Video Games Star Review', dataframe=train_review_data, x_axis='overall', y_axis='vader_compound', ax=ax[0][2])
    d_analysis.validation_scoring(title='Roberta\'s Positive Score by Video Games Star Review', dataframe=train_review_data, x_axis='overall', y_axis='roberta_positive', ax=ax[1][0])
    d_analysis.validation_scoring(title='Roberta\'s Negative Score by Video Games Star Review', dataframe=train_review_data, x_axis='overall', y_axis='roberta_negative', ax=ax[1][1])
    d_analysis.validation_scoring(title='Roberta\'s Neutral Score by Video Games Star Review', dataframe=train_review_data, x_axis='overall', y_axis='roberta_neutral', ax=ax[1][2])
    plt.tight_layout()
    plt.show()