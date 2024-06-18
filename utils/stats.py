"""
Module for computing the statistics of the dataset.
"""

import sys
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from collections import Counter
from wordcloud import WordCloud
sys.path.append("../model/")
from models import load_data # type: ignore

logging.basicConfig(level=logging.INFO, filemode='a', filename='stats.log')
nltk.download('punkt')
nltk.download('stopwords')



def len_per_column(data: pd.DataFrame):
    """
    Add new columns with some information.
    """
    try:
        data['num_words'] = data['Comments'].apply(lambda x: len(x.split()))
        data['num_unique_words'] = data['Comments'].apply(lambda x: len(set(x.split())))
        plt.figure(figsize = (8, 6))
        sns.histplot(data['num_words'], bins=5)
        plt.title('Words per Distribution')
        plt.show(), data['num_words'].skew()
    except Exception as e:
        logging.info("Exception occurred on len column! %s", str(e))


def label_stats(data: pd.DataFrame):
    """
    On the label column, perform some statistics.
    """
    try:
        if 'label' in data.columns:
            sentiment_counts = data['label'].value_counts()
            logging.info(f"The number in sentiments{sentiment_counts}")
            sentiment_stats = data['label'].describe()
            plt.figure(figsize=(12, 6))
            sentiment_counts.plot(kind="pie", autopct='%1.2f%%')    
            plt.title('Sentiment Distribution', loc='center')
            plt.axis('off')
            plt.show()
            logging.info("Sentiment Statistics:")
            logging.info(sentiment_stats)
    except Exception as e:
        logging.info("Exception occurred on label stats! %s", str(e))


def wordcloud_graphic(all_words: str):
    """
    Plot a wordcloud.
    """
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show()
        total_words = len(all_words.split())
        total_unique_words = len(list(set(all_words.split())))
        return total_words, total_unique_words
    except Exception as e:
        logging.info("Exception occured on wordcloud! %s", str(e))
        return None


def most_common_words(all_words: str):
    """
    Load the most common words (could be stopwords).
    """
    try:
        word_counts = Counter(all_words.split())
        common_words = word_counts.most_common(20)
        words_df = pd.DataFrame(common_words, columns=['word', 'count'])
        plt.figure(figsize=(12, 6))
        sns.barplot(x='count', y='word', data=words_df)
        plt.title('Top 20 Most Frequent Words')
        plt.show()
        return common_words
    except Exception as e:
        logging.info("Exception occurred on common words! %s", str(e))
        return None


def main():
    """
    Assemble everything from here!
    """
    try:
        path = '../data/'
        data = load_data(path + 'Siswati_Sentiment.csv')
        len_per_column(data=data)
        label_stats(data=data)
        all_words = ' '.join(data['Comments'])
        total_words, total_unique_words = wordcloud_graphic(all_words=all_words)
        logging.info("The total number of words is: {}".format(total_words))
        logging.info("The total number of unique words is: {}".format(total_unique_words))
    except Exception as e:
        logging.info("Unhandled Exception! %s", str(e))


if __name__ == "__main__":
    main()
