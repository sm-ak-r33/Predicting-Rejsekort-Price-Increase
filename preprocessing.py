import os
import warnings
import sys
import logging
import pandas as pd
import numpy as np

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    xlsx_url = (
        "https://github.com/sm-ak-r33/Predicting-Rejsekort-Price-Increase-2023/raw/refs/heads/main/Data.xlsx"
    )
    try:
        data = pd.read_excel(xlsx_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test excel, check your internet connection. Error: %s", e
        )

    # Read the wine-quality csv file from the URL
    updated_xlsx_url = (
        "https://github.com/sm-ak-r33/Predicting-Rejsekort-Price-Increase-2023/raw/refs/heads/main/Data(update).xlsx"
    )
    try:
        data_update = pd.read_excel(updated_xlsx_url)
    except Exception as e:
        logger.exception(
            "Unable to download training & test excel, check your internet connection. Error: %s", e
        )

def clean_dataframe(df):
    """
    Removes the index from the DataFrame and renames specific columns.

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    # Remove the index by resetting it
    df = df.reset_index(drop=True)

    # Rename columns
    rename_mapping = {
        'Afgangsdato År - Måned - Dato': 'date',
        'Antal Personrejser': 'passengers'
    }
    df = df.rename(columns=rename_mapping)

    # Define the regex pattern for dd-mm-yyyy
    pattern = r'^\d{2}-\d{2}-\d{4}$'

    # Filter rows that match the pattern
    df = df[df['date'].str.match(pattern, na=False)]

    df['date']=pd.to_datetime(df['date'], dayfirst=True)

    return df


def append_and_update(df1, df2):
    """
    Appends `df2` to `df1` while updating the 'passengers' column for the rows
    in `df1` that have a matching 'date' in `df2`, without creating duplicate 'passengers' columns.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame (updated results).
    
    Returns:
        pd.DataFrame: The updated DataFrame with `df1`'s missing dates updated from `df2`.
    """

    # Merge the DataFrames on 'date' using an outer join
    merged_df = pd.merge(df1, df2, on='date', how='outer', suffixes=('_df1', '_df2'))

    # Update the 'passengers' column, preferring values from df2
    merged_df['passengers'] = merged_df['passengers_df2'].combine_first(merged_df['passengers_df1'])

    # Drop the intermediate columns
    merged_df = merged_df.drop(columns=['passengers_df1', 'passengers_df2'])

    return merged_df

df1=clean_dataframe(data)
df2=clean_dataframe(data_update)
df=append_and_update(df1, df2)

df.to_csv('data_cleaned.csv',index=False)