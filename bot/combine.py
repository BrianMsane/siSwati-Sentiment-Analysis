"""
Module for combining collected comments.
"""

import os
import logging
import pandas as pd

logging.basicConfig(level = logging.INFO)

def main():
    """
    Combine all csv files.
    """

    try:

        cwd = os.getcwd()
        csv_files = [file for file in os.listdir(cwd) if file.endswith('.csv')]
        dfs = []

        for file in csv_files:
            file_path = os.path.join(cwd, file)
            df = pd.read_csv(file_path)
            dfs.append(df)

        combined_df = pd.concat(dfs, ignore_index = True)
        combined_df.to_csv("Final_dataset.csv", index = False)

    except Exception as e:
        logging.info("Exception while combining files: %s", str(e))


if __name__ == "__main__":
    main()
