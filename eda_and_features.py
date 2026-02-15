import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda_and_feature_engineering(filepath):
    print("Loading cleaned dataset...")
    df = pd.read_csv(filepath)
    
    # 1. Correlation Analysis (Identifying Multicollinearity)
    print("\n--- Correlation Matrix ---")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    print("Correlation heatmap saved as 'correlation_heatmap.png'. Review this image.")
    
    # 2. Outlier Treatment using IQR (Interquartile Range) Method
    print("\n--- Outlier Treatment ---")
    # We will cap outliers rather than drop them to preserve data volume.
    features_to_cap = ['views', 'likes', 'comments', 'watch_time_minutes', 'subscribers']
    
    for col in features_to_cap:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Calculate how many outliers exist
        outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        print(f"{col}: Found {outliers_count} outliers. Capping them...")
        
        # Cap the outliers
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])

    # 3. Categorical Encoding (Preparing for Modeling)
    print("\n--- Categorical Encoding ---")
    # Linear Regression cannot process text like 'Mobile' or 'Gaming'. 
    # We use One-Hot Encoding and drop the first category to avoid the Dummy Variable Trap.
    categorical_cols = ['category', 'device', 'country']
    print(f"Encoding columns: {categorical_cols}")
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Save the final processing step
    final_filepath = 'model_ready_data.csv'
    df_encoded.to_csv(final_filepath, index=False)
    print(f"\nFinal feature-engineered data saved to: {final_filepath}")
    print(f"New shape after encoding: {df_encoded.shape}")

if __name__ == "__main__":
    # Ensure this runs on the file generated in Phase 1
    perform_eda_and_feature_engineering('C:/Users/kubad/Desktop/PROJECTS/Guvi_project-3/cleaned_youtube_data.csv')