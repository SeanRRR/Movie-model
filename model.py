import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np


def create_model(input_data):
    num_features = input_data.shape[1]
    model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, max_depth=6, random_state=42)
    optimizer = None  # Not required for gradient boosting
    return model, optimizer


def preprocess_data(df):
    # Selecting relevant columns
    COLUMN_NAMES = [
        "Series_Title", "Released_Year", "Certificate", "Runtime", "Genre", "IMDB_Rating", "Meta_score",
        "Director", "Star1", "Star2", "Star3", "Star4", "No_of_Votes", "Gross", "Budget"
    ]
    df = df[COLUMN_NAMES].dropna()

    # Convert numerical columns
    df["Released_Year"] = df["Released_Year"].astype(int)
    df["IMDB_Rating"] = df["IMDB_Rating"].astype(float)
    df["Meta_score"] = df["Meta_score"].astype(float)
    df["No_of_Votes"] = df["No_of_Votes"].astype(int)
    df["Gross"] = df["Gross"].replace({',': ''}, regex=True).astype(float)
    df["Budget"] = df["Budget"].replace({',': ''}, regex=True).astype(float)
    df["Runtime"] = df["Runtime"].str.replace(" min", "").astype(int)

    # One-hot encoding for categorical variables
    categorical_cols = ["Certificate", "Genre", "Director", "Star1", "Star2", "Star3", "Star4"]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_cats = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_cols))

    # Concatenate encoded categorical features and numerical features
    df = df.drop(columns=categorical_cols)
    df = pd.concat([df, encoded_df], axis=1)

    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ["Released_Year", "IMDB_Rating", "Meta_score", "No_of_Votes", "Gross", "Budget", "Runtime"]
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return torch.tensor(df.values, dtype=torch.float32), torch.tensor(df["Gross"].values, dtype=torch.float32)
