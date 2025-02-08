import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from prep_data import get_prepared_data
from model import create_model


# Wrapper to make Gradient Boosting behave like a PyTorch model
class SklearnModelWrapper(torch.nn.Module):
    def __init__(self, sklearn_model):
        super().__init__()
        self.sklearn_model = sklearn_model

    def forward(self, x):
        x_np = x.detach().numpy()  # Convert Tensor to NumPy
        pred = self.sklearn_model.predict(x_np)  # Predict with Gradient Boosting
        return torch.tensor(pred, dtype=torch.float32).unsqueeze(1)  # Convert back to Tensor


def train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val, training_updates=True):
    # Handle Gradient Boosting separately
    if optimizer is None:
        model.fit(X_train.numpy(), y_train.numpy().ravel())  # Train Gradient Boosting
        return SklearnModelWrapper(model)  # Wrap so `main.py` can call model(X_test)

    # PyTorch Training Loop
    num_epochs = 1000
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        if training_updates and epoch % (num_epochs // 10) == 0:
            with torch.no_grad():
                output = model(X_val)
                val_loss = criterion(output, y_val)
                print(f"Epoch {epoch} | Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}")

    return model


if __name__ == '__main__':
    # Load data
    features, target = get_prepared_data()

    # Create training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2)

    # Define model
    model, optimizer = create_model(X_train)

    # Define loss function (only used for PyTorch)
    criterion = nn.MSELoss() if optimizer is not None else None

    # Train model
    model = train_model(model, optimizer, criterion, X_train, y_train, X_val, y_val)

    # If model is PyTorch, save weights
    if optimizer is not None:
        torch.save(model, "saved_weights/model.pth")
        print("Model saved as model.pth")
