import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np


class GradientBoostingModel(nn.Module):
    def __init__(self, input_dim):
        super(GradientBoostingModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)


def create_model(input_data):
    """Creates and returns a PyTorch gradient boosting model and an optimizer."""
    num_features = input_data.shape[1]
    model = GradientBoostingModel(num_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # You can adjust the learning rate
    return model, optimizer


def preprocess_data(df):
    """Preprocesses the dataset: handles missing values, encodes categorical data, and scales numerical features."""

    cols_to_keep = [
        "Series_Title", "Released_Year", "Certificate", "Runtime", "Genre",
        "IMDB_Rating", "Meta_score", "Director", "Star1", "Star2", "Star3",
        "Star4", "No_of_Votes", "Gross", "Budget"
    ]

    df = df[cols_to_keep].dropna()

    try:
        df["Released_Year"] = df["Released_Year"].astype(int)
        df["IMDB_Rating"] = df["IMDB_Rating"].astype(float)
        df["Meta_score"] = df["Meta_score"].astype(float)
        df["No_of_Votes"] = df["No_of_Votes"].astype(int)
        df["Gross"] = df["Gross"].replace({',': ''}, regex=True).astype(float)
        df["Budget"] = df["Budget"].replace({',': ''}, regex=True).astype(float)
        df["Runtime"] = df["Runtime"].str.replace(" min", "").astype(int)
    except ValueError as e:
        print(f"Data conversion error: {e}")
        return None

    categorical_cols = ["Certificate", "Genre", "Director", "Star1", "Star2", "Star3", "Star4"]
    encoded_dfs = [pd.get_dummies(df[col], prefix=col) for col in categorical_cols]
    df = df.drop(columns=categorical_cols).reset_index(drop=True)
    df = pd.concat([df] + encoded_dfs, axis=1)

    numerical_cols = ["Released_Year", "IMDB_Rating", "Meta_score", "No_of_Votes", "Gross", "Budget", "Runtime"]
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

    X = torch.tensor(df.drop(columns=["Gross"]).values, dtype=torch.float32)
    y = torch.tensor(df["Gross"].values, dtype=torch.float32).view(-1, 1)

    return X, y
