"""
Module for cleaning the dataset.
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, filemode='a', filename='clean.log')


class DatasetPreprocessor:
    """
    Class for cleaning raw dataset.
    """

    def clean_dataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all cleaning methods to the dataset.

        Parameters:
        ----------
        data : pd.DataFrame
            The dataset to be cleaned.

        Returns:
        -------
        pd.DataFrame
            The cleaned dataset, or None if an error occurs.
        """

        try:

            data = self.unique_id(data)
            data = self.rename_columns(data)
            data = self.remove_duplicates(data)
            return data

        except Exception as e:
            logging.info("Exception while cleaning dataset! %s", str(e))
            return None

    def unique_id(self, data: pd.DataFrame) -> pd.DataFrame:
        data.reset_index(inplace=True)
        data.rename(columns={'index': 'ID'}, inplace=True)
        return data

    def remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        data.drop_duplicates(inplace=True)
        return data

    def rename_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        data.rename(columns={'text': 'Comments', 'Sentiment': 'label'}, inplace=True)
        return data
