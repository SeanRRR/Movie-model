import pandas as pd
import numpy as np
import requests
import torch
import time

# Load dataset and return cleaned and preprocessed features and target
# TODO: use more of the columns, preprocess them in a different way,
#       or even include new features (e.g. from other datasets)

# columns: 'Poster_Link', 'Series_Title', 'Released_Year', 'Certificate',
#          'Runtime', 'Genre', 'IMDB_Rating', 'Overview', 'Meta_score', 'Director',
#          'Star1', 'Star2', 'Star3', 'Star4', 'No_of_Votes'

# columns used in template: 'Released_Year', 'Certificate', 'Runtime', 'Genre' (kind of',
# 'IMDB_Rating', 'Meta_score', 'No_of_Votes'
def get_prepared_data(data_path="data"):

    # Load raw data pause
    # this function tries to combine all .csv files in the data folder
    # it matches them up using the "Series_Title" column
    # if you want to use additional datasets, make sure they have a "Series_Title" column
    # if not, you will need additional logic to join the datasets
    # do not rename the column by hand, add code before this point to rename it
    # remember: we will not manually modify your datasets, so your code must do any formatting automatically
    data = get_raw_data(data_path)
    # Drop columns in text format (not used in the demo, may be useful to you)
    data = data.drop(columns=["Poster_Link", "Series_Title", "Overview", "Director", "Star1", "Star2", "Star3", "Star4"])
    # take only the first genre from the list of genres (you might want to do something more sophisticated)
    data["Genre"] = data["Genre"].apply(lambda x: x.split(",")[0])

    # convert "Gross" into a number (remove ",")
    data["Gross"] = data["Gross"].apply(lambda x: int(x.replace(",", ""))
                                        if type(x) == str else x)
    # Convert categorical columns to one-hot encoding
    data = pd.get_dummies(data, dtype=int)
    # Define features and target
    features = data.drop(columns=["Gross"])
    target = data["Gross"]
    # Convert to numpy arrays
    features = np.array(features)

    target = np.array(target).reshape(-1, 1)

    # Convert to torch tensors
    features = torch.tensor(features, dtype=torch.float32)
    target = torch.tensor(target, dtype=torch.float32)

    test_budget = get_movie_budget("The Godfather", 1972)
    print(test_budget)

    # Adding the Budget column
    df['Budget'] = df.apply(lambda row: get_movie_budget(row['Series_Title'], row['Released_Year']) if pd.notnull(row['Released_Year']) else None, axis=1)
    # Save updated dataset

    df.dropna()
    df_filtered = df[df['Budget'] != 0]
    print(df_filtered)
    df_filtered.to_csv('data/imdb_top_1000.csv', index=False)

    print("Dataset updated with budget information.")

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

def get_all_titles(data_path="data"):
    data = get_raw_data(data_path)
    return data["Series_Title"]

def get_raw_data(path="data"):
    # read in every csv file, join on "Series_Title"
    # return the raw data
    import os
    files = os.listdir(path)
    data = pd.DataFrame()
    for file in files:
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(path, file))
            df = df.dropna()
            # join on "Series_Title"
            if data.empty:
                data = df
            else:
                data = data.merge(df, on="Series_Title")
    return data