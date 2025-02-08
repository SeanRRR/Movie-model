import pandas as pd
import numpy as np
import requests
import torch
import time
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import OneHotEncoder


# Function to get prepared data
def get_prepared_data(data_path="data", k_best=100):
    # Load raw data
    data = get_raw_data(data_path)

    # Add the 'Budget' column
    if "Budget" not in data.columns:
        data["Budget"] = data.apply(
            lambda row: get_movie_budget(row["Series_Title"], row["Released_Year"]) if pd.notnull(
                row["Released_Year"]) else None,
            axis=1
        )
    # Drop rows with missing or invalid 'Budget'
    data.dropna(subset=["Budget"], inplace=True)
    data = data[data["Budget"] != 0]

    # Drop unnecessary columns
    data = data.drop(columns=["Poster_Link","Overview", "Star2", "Star3", "Star4"],
                     errors='ignore')

    # Take only the first genre
    data["Genre"] = data["Genre"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else x)

    # Convert "Gross" into a number
    data["Gross"] = data["Gross"].apply(lambda x: int(x.replace(",", "")) if isinstance(x, str) else x)

    # Apply OneHotEncoder to categorical columns
    categorical_columns = ["Certificate", "Genre", "Director"]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    encoded_df.to_csv("testoutput.csv", index=False)

    # Drop original categorical columns and concatenate encoded features
    data = data.drop(columns=categorical_columns)
    data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

    print("Dataset updated with budget information and K best features selected.")

    # Define features and target
    features = data.drop(columns=["Gross"], errors='ignore')
    target = data["Gross"]

    # Convert to numpy arrays ensuring all features are numeric
    features = features.select_dtypes(include=[np.number])
    features = np.array(features, dtype=np.float32)
    target = np.array(target).reshape(-1, 1).astype(np.float32)

    # Select K best features
    selector = SelectKBest(score_func=mutual_info_regression, k=min(k_best, features.shape[1]))
    selected_features = selector.fit_transform(features, target.ravel())

    # Convert to torch tensors
    features = torch.tensor(selected_features, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    return features, target


# Load dataset
df = pd.read_csv('data/imdb_top_1000.csv')

# Your TMDB API key
TMDB_API_KEY = '1022c77fe927cff4f34f59e88869c4f7'

# Base URL for TMDB search and movie details
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
DETAILS_URL = "https://api.themoviedb.org/3/movie/{}"


# Function to get budget from TMDB
def get_movie_budget(title, year):
    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "year": year,
    }
    response = requests.get(SEARCH_URL, params=params).json()
    results = response.get("results", [])
    if results:
        movie_id = results[0]["id"]
        details_response = requests.get(DETAILS_URL.format(movie_id), params={"api_key": TMDB_API_KEY}).json()
        return details_response.get("budget", None)
    return None


# Function to get all titles
def get_all_titles(data_path="data"):
    data = get_raw_data(data_path)
    return data["Series_Title"]


# Function to get raw data
def get_raw_data(path="data"):
    import os
    files = os.listdir(path)
    data = pd.DataFrame()
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna()
            if data.empty:
                data = df
            else:
                data = data.merge(df, on="Series_Title")
    return data
