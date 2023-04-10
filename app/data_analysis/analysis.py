'''Import Libraries'''
import seaborn as sns
import matplotlib.pyplot as plt

class DataAnalysis():

    def understanding_data(self, dataframe: object) -> None:
        print(f'Names of features are:\n{dataframe.columns}')
        print(f'\nTop 5 records of Data:\n{dataframe.head(10)}')
        print(f'\nNull values:\n{dataframe.isna().sum()}')
        print(f'\nType of data types of each features:\n{dataframe.dtypes}')
        print(f'\nMeasure of Central Tendency and Dispersion:\n{dataframe.describe()}')
        print(f'\nValue count of Overall ratings:\n{dataframe.overall.value_counts()}')

    def overall_rating_distribution(self, dataframe: object) -> None:
        sns.countplot(data=dataframe, x='overall')
        plt.show()

    def validation_scoring(self, title: str, dataframe: object, x_axis: str, y_axis: str, ax: tuple = None):
        sns.barplot(data=dataframe, x=x_axis, y=y_axis, ax=ax).set_title(title)
        