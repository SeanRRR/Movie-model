import pandas as pd
import numpy as np
import requests
import torch
import time
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
import os


# Function to load and prepare data
def get_prepared_data(data_path="data", k_best=100):
    """
    Loads the dataset, processes it, and returns selected features as PyTorch tensors.

    Args:
        data_path (str): Path to the dataset.
        k_best (int): Number of best features to keep.

    Returns:
        torch.Tensor, torch.Tensor: Processed features and target variable.
    """

    print("Loading raw data...")
    data = get_raw_data(data_path)

    # Check if 'Budget' column is missing, then fetch it using API
    if "Budget" not in data.columns:
        print("Fetching missing budget data...")
        data["Budget"] = data.apply(lambda row: get_movie_budget(row["Series_Title"], row["Released_Year"])
        if pd.notnull(row["Released_Year"]) else None, axis=1)

    # Remove rows where Budget is missing or invalid
    data.dropna(subset=["Budget"], inplace=True)
    data = data[data["Budget"] != 0]

    # Drop unnecessary columns (some may not exist)
    data = data.drop(columns=["Poster_Link", "Overview", "Star2", "Star3", "Star4"], errors='ignore')

    # Keep only the first genre listed (since some movies have multiple genres)
    data["Genre"] = data["Genre"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else x)

    # Convert 'Gross' into a number (some values may contain commas)
    def clean_gross_value(value):
        if isinstance(value, str):
            try:
                return int(value.replace(",", ""))
            except ValueError:
                return np.nan
        return value

    data["Gross"] = data["Gross"].apply(clean_gross_value)
    data.dropna(subset=["Gross"], inplace=True)  # Remove any rows where Gross is still NaN

    # Encode categorical columns
    categorical_columns = ["Certificate", "Genre", "Director"]
    print("Encoding categorical features...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_features = encoder.fit_transform(data[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

    # Save encoded features for checking
    encoded_df.to_csv("test_output.csv", index=False)

    # Drop original categorical columns and merge with encoded ones
    data = data.drop(columns=categorical_columns)
    data = pd.concat([data.reset_index(drop=True), encoded_df], axis=1)

    print("Dataset cleaned and categorical features encoded.")

    # Define features and target
    features = data.drop(columns=["Gross"], errors='ignore')
    target = data["Gross"]

    # Ensure all features are numeric
    features = features.select_dtypes(include=[np.number])

    # Convert to numpy arrays
    features = np.array(features, dtype=np.float32)
    target = np.array(target, dtype=np.float32).reshape(-1, 1)  # Ensure correct shape

    # Select best features (make sure k is not greater than available features)
    k_best = min(k_best, features.shape[1])
    print(f"Selecting top {k_best} features...")
    selector = SelectKBest(score_func=mutual_info_regression, k=k_best)
    selected_features = selector.fit_transform(features, target.ravel())

    # Convert to PyTorch tensors
    features_tensor = torch.tensor(selected_features, dtype=torch.float32)
    target_tensor = torch.tensor(target, dtype=torch.float32)

    print("Feature selection complete. Returning tensors.")

    return features_tensor, target_tensor


# Function to get budget data from TMDB API
def get_movie_budget(title, year):
    """
    Fetches the budget for a movie from TMDB API.

    Args:
        title (str): Movie title.
        year (int): Release year.

    Returns:
        int or None: Budget in dollars, or None if not found.
    """

    params = {
        "api_key": TMDB_API_KEY,
        "query": title,
        "year": year,
    }

    try:
        response = requests.get(SEARCH_URL, params=params).json()
        results = response.get("results", [])
        if results:
            movie_id = results[0]["id"]
            details_response = requests.get(DETAILS_URL.format(movie_id), params={"api_key": TMDB_API_KEY}).json()
            return details_response.get("budget", None)
    except Exception as e:
        print(f"Error fetching budget for {title}: {e}")

    return None


# Function to load raw dataset from CSV files
def get_raw_data(path="data"):
    """
    Reads all CSV files in the given directory and merges them.

    Args:
        path (str): Directory containing CSV files.

    Returns:
        pd.DataFrame: Merged dataset.
    """

    files = os.listdir(path)
    data = pd.DataFrame()

    for file in files:
        if file.endswith(".csv"):
            print(f"Loading file: {file}")
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna()  # Drop missing values (simpler approach)

            if data.empty:
                data = df
            else:
                try:
                    data = data.merge(df, on="Series_Title")
                except Exception as e:
                    print(f"Error merging {file}: {e}")

    return data


# API key and URLs
TMDB_API_KEY = '1022c77fe927cff4f34f59e88869c4f7'
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
DETAILS_URL = "https://api.themoviedb.org/3/movie/{}"

# Load dataset
df = pd.read_csv('data/imdb_top_1000.csv')
