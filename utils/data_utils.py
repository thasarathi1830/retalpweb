import pandas as pd

def clean_data(df):
    # Example data cleaning
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    df.columns = df.columns.str.strip()
    return df

