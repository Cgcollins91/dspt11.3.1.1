import numpy as np
import pandas as pd


class Helper:
    def __init__(self, df_in):
        self.df = df_in
        self.n = len(df_in.index)

    def null_count(self):
        df = self.df
        return df.isnull().sum().sum()

    def train_test_split(self, frac):
        df = self.df
        n = self.n
        train_n = round(n * frac)
        train_df = df.iloc[:train_n, :]
        test_df = df.iloc[train_n:, :]
        return train_df, test_df

# Test helper functions perform intended function:
df = pd.read_csv('/Users/chriscollins/Documents/Datasets/Kaggle Advertising/train.csv')

df_helper = Helper(df)
print(df_helper.null_count())
print(df.isnull().sum().sum())
train, test = df_helper.train_test_split(.3)

print(train.shape, test.shape)
