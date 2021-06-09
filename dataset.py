import pandas as pd


def get_dataset():
    #link = "https://github.com/thiagodiasbispo/intro_colab_pycharm/raw/main/data/imdb_sentiment_10k.csv"
    file = "data/imdb_sentiment_10k.csv"
    df = pd.read_csv(file)
    return df