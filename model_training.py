import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

def train_and_evaluate():
    print("Loading model-ready data...")
    try:
        df = pd.read_csv('model_ready_data.csv')
    except FileNotFoundError:
        print("Error: 'model_ready_data.csv' not found. Did you finish Phase 2?")
        return

    # 1. Separate Features (X) and Target (y)
    # The target is what we want to predict. The features are the inputs.
    X = df.drop(columns=['ad_revenue_usd'])
    y = df['ad_revenue_usd']

    # 2. Train-Test Split
    # We hide 20% of the data from the model during training to test its actual predictive power later.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # 3. Model Initialization and Training
    print("\nTraining Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 4. Predictions on the unseen Test Set
    predictions = model.predict(X_test)

    # 5. Brutal Evaluation
    # R²: How much variance in revenue is explained by your features? (1.0 is perfect, 0 is garbage)
    # RMSE: On average, how many dollars are your predictions off by, penalizing large errors?
    # MAE: On average, how many dollars are your predictions off by, treating all errors equally?
    r2 = r2_score(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)

    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     ${rmse:.2f}")
    print(f"MAE:      ${mae:.2f}")

    if r2 < 0.5:
        print("\nWARNING: Your R² is abysmal. Your features barely explain the revenue. Linear regression might be too weak for this dataset.")
    elif r2 > 0.95:
        print("\nWARNING: Your R² is suspiciously high. Check for data leakage; you might be predicting the past.")

    # 6. Save the Model and Feature Columns for Streamlit
    # Streamlit needs to know exactly what columns the model expects, in the exact same order.
    model_data = {
        'model': model,
        'features': X.columns.tolist()
    }
    
    with open('revenue_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
        
    print("\nModel and feature schema saved to 'revenue_model.pkl'")

if __name__ == "__main__":
    train_and_evaluate()