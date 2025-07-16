import pandas as pd

# load the dataset

df = pd.read_csv('churn_data.csv')

# show basic info
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())