import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import drive


# Load preprocessed data
data_path = '/content/drive/MyDrive/Autofill_App/Processed_Data/enhanced_data_preprocessed.csv'
df = pd.read_csv(data_path)

# Define features and target
feature_cols = ['framework_encoded', 'domain_encoded', 'current_average'] + [f'gap_{i}' for i in range(100)]
X = df[feature_cols]
y = df['current_maturity']

# Set up MLflow
mlflow.set_tracking_uri('file:///content/mlruns')  # Local tracking in Colab
mlflow.set_experiment('compliance_maturity_prediction')

# Define model and hyperparameters
model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None]
}

# Perform grid search and log with MLflow
with mlflow.start_run():
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    # Log parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log metrics
    rmse = np.sqrt(-grid_search.best_score_)
    mlflow.log_metric('rmse', rmse)
    cv_r2 = cross_val_score(grid_search.best_estimator_, X, y, cv=5, scoring='r2')
    mlflow.log_metric('r2_mean', np.mean(cv_r2))
    
    # Log model
    mlflow.sklearn.log_model(grid_search.best_estimator_, 'model')
    
    # Save model locally for deployment
    model_path = '/content/drive/MyDrive/Autofill_App/Models/compliance_model'
    mlflow.sklearn.save_model(grid_search.best_estimator_, model_path)
    
    print(f"Best RMSE: {rmse:.2f}, R2: {np.mean(cv_r2):.2f}")
    print(f"Model saved to: {model_path}")
