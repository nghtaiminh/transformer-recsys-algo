from .base import AbstractDataset
import pandas as pd
from datetime import date

class AmazonBooksDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'AmazonBooks'

    @classmethod
    def url(cls):
        return 'https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall/Books.csv'

    @classmethod
    def zip_file_content_is_folder(cls):
        return False

    def load_ratings_df(self):
        file_path = 'Data/AmazonBooks/ratings.csv'
        df = pd.read_csv(file_path, header=None)
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df


