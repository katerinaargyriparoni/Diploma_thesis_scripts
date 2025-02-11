import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
from itertools import combinations

# Load data from CSV with ';' as delimiter
df = pd.read_csv(
    '/Users/katerinaargyriparoni/Desktop/Diploma_thesis/dataset_1/final/Final_Interpolated_Data_2020_2024_22_11_2024.csv',
    delimiter=';')

# Define features (X) and target (y)
X = df[['EVI', 'NDBI', 'NDVI', 'SAVI', 'SMI', 'NO2', 'O3', 'PM10', 'PM2.5', 'urban_mean', 'rural_mean', 'urban_max',
        'temp_rural', 'mrt']]
# y = df['UHI']
y = (X['urban_mean']-X['rural_mean']/np.std(X['rural_mean']))
# Normalize X using Min-Max scaling
X = (X - X.min()) / (X.max() - X.min())

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of features for combination
features = ['EVI', 'NDBI', 'NDVI', 'SAVI', 'SMI', 'NO2', 'O3', 'PM10', 'PM2.5',
            # 'urban_mean', 'rural_mean',
            'urban_max',
            'temp_rural', 'mrt']

# List to store results
results = []

# Iterate over all possible feature combinations
for r in range(10, len(features) + 1):
    for combo in combinations(features, r):
        # Select subset of features
        x_train_subset = x_train[list(combo)]
        x_test_subset = x_test[list(combo)]

        # Models to train and evaluate
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(max_depth=2),
            'Support Vector Regression': SVR(),
            'Lasso Regression': Lasso(alpha=0.2),
            'Random Forest': RandomForestRegressor(n_estimators=150, criterion='absolute_error'),
            'XGBoost': xgb.XGBRegressor(
                objective='reg:squarederror',
                learning_rate=0.005,
                n_estimators=1000,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'MLP Regressor': MLPRegressor(hidden_layer_sizes=1000, random_state=1, max_iter=5000, batch_size=32,
                                          early_stopping=True)
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            model.fit(x_train_subset, y_train)
            preds = model.predict(x_test_subset)

            # Calculate evaluation metrics
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mae = mean_absolute_error(y_test, preds)
            r2 = r2_score(y_test, preds)

            # Store results
            results.append({
                'Model': model_name,
                'Features': combo,
                'RMSE': rmse,
                'MAE': mae,
                'R²': r2
            })
            print(f"{model_name} with features {combo} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('model_evaluation_results.csv', index=False)

print("\nAll results have been saved to 'model_evaluation_results.csv'.")