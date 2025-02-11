import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


class MyModel(nn.Module):
    """A simple neural network for predicting movie revenue."""

    def __init__(self, input_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # First layer with 64 neurons
        self.fc2 = nn.Linear(64, 32)  # Second layer with 32 neurons
        self.output_layer = nn.Linear(32, 1)  # Final output layer
        self.dropout = nn.Dropout(0.2)  # Helps prevent overfitting

    def forward(self, x):
        """Defines how the input flows through the network."""
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = torch.relu(self.fc2(x))
        return self.output_layer(x)  # No activation for regression output


def create_model(data, lr=0.003):
    """
    Creates a model and optimizer.

    Args:
        data (pd.DataFrame): The input data.
        lr (float): Learning rate.

    Returns:
        model, optimizer
    """
    feature_count = data.shape[1]
    model = MyModel(feature_count)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def preprocess_movie_data(df):
    """
    Prepares movie data for training.

    Args:
        df (pd.DataFrame): Raw dataset.

    Returns:
        torch.Tensor, torch.Tensor: Processed features and target.
    """

    # Keeping only the relevant columns
    cols_needed = [
        "Released_Year", "IMDB_Rating", "Meta_score", "No_of_Votes",
        "Gross", "Budget", "Runtime", "Certificate", "Genre",
        "Director", "Star1", "Star2", "Star3", "Star4"
    ]

    df = df[cols_needed].copy()  # Make a copy to avoid modifying the original

    # Filling missing values in numeric columns
    numeric_cols = ["Released_Year", "IMDB_Rating", "Meta_score", "No_of_Votes", "Gross", "Budget", "Runtime"]
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())  # Fill NaNs with mean values

    # Convert numerical columns to proper types
    df["Released_Year"] = df["Released_Year"].astype(int)
    df["IMDB_Rating"] = df["IMDB_Rating"].astype(float)
    df["Meta_score"] = df["Meta_score"].astype(float)
    df["No_of_Votes"] = df["No_of_Votes"].astype(int)

    # Some budget and gross values have commas, need to clean them
    df["Gross"] = df["Gross"].astype(str).str.replace(",", "").astype(float)
    df["Budget"] = df["Budget"].astype(str).str.replace(",", "").astype(float)

    # One-hot encoding for categorical features
    categorical_cols = ["Certificate", "Genre", "Director", "Star1", "Star2", "Star3", "Star4"]
    df = pd.get_dummies(df, columns=categorical_cols)

    # Normalize numerical values
    for col in numeric_cols:
        df[col] = (df[col] - df[col].mean()) / df[col].std()  # Standardization

    # Separate target variable
    y = df["Gross"]
    X = df.drop(columns=["Gross"])

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)  # Reshape for PyTorch

    return X_tensor, y_tensor
