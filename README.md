# IMDB Box Office Predictor, GDG Mini-ML Competition

### Description
Predicts US/CA box office earnings for the top 1000 IMDB movies using a PyTorch MLP. Built in 7 days for the GDG Mini-ML Competition — achieved R² of 60.4% via 5-fold cross-validation. Won third place!

### Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Data Science:** Pandas, NumPy, Scikit-learn
* **Version Control:** Git

### Model Details
Architecture: Multi-Layer Perceptron (MLP)
* **Layer 1:** 64 neurons (ReLU)
* **Dropout:** 0.2 (to prevent overfitting within a small dataset N = 1000)
* **Layer 2:** 32 neurons (ReLU)
* **Output:** 1 neuron (Linear)

Preprocessing: We performed One-Hot Encoding on categorical features (Genre, Director, Stars) and Standardization (scaling to mean=0, std=1) on numerical features like Budget and Runtime. Missing values were handled by filling them with the mean of the column.

Training Strategy: Trained using the Adam Optimizer with a learning rate of 0.003 and Mean Squared Error (MSE) as the loss function.

Cross-Validation: 5-fold cross-validation yielded an average R² Score of 60.4%.

### How to Run
```bash
# Install dependencies
pip install torch pandas numpy scikit-learn

# Data preprocessing
python prep_data.py

# Execution and validation
python main.py
```
