import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# Load data
df = pd.read_csv(
    '/Users/katerinaargyriparoni/Downloads/Interpolated_daytime_nighttime_dataset_2020_2024_final_ 2 αντίγραφο.csv',
    delimiter=';')

# Remove outliers based on 'UHI' column
#q1 = df['UHI'].quantile(0.28)
#q3 = df['UHI'].quantile(0.48)
q1 = df['UHI'].quantile(0.28)
q3 = df['UHI'].quantile(0.48)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

df['outlier'] = (df['UHI'] < lower_bound) | (df['UHI'] > upper_bound)

# Plot UHI values with outliers in orange and inliers in blue
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['UHI'], c=np.where(df['outlier'], 'orange', 'blue'), alpha=0.7)
plt.axhline(y=lower_bound, color='red', linestyle='--', label=f'Lower Bound ({lower_bound:.2f})')
plt.axhline(y=upper_bound, color='green', linestyle='--', label=f'Upper Bound ({upper_bound:.2f})')
plt.title("UHI Values with Outliers Highlighted")
plt.xlabel("Index")
plt.ylabel("UHI Value")
plt.legend()
plt.show()

# Remove outliers from the dataset
df = df[~df['outlier']]
df = df.drop(columns=['outlier'])

# Reshape UHI column to a 2D array for KMeans
uhi_values = df['UHI'].values.reshape(-1, 1)

# Define the number of clusters (classes) for K-means
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans and predict cluster labels
df['uhi_class'] = kmeans.fit_predict(uhi_values)

# Sort the clusters by their mean UHI value
cluster_order = df.groupby('uhi_class')['UHI'].mean().sort_values().index
cluster_mapping = {old: new for new, old in enumerate(cluster_order)}

# Map the clusters to ordered class labels
df['uhi_class'] = df['uhi_class'].map(cluster_mapping)

# Display the mean UHI for each cluster
uhi_cluster_summary = df.groupby('uhi_class')['UHI'].agg(['min', 'max', 'mean'])

# Get the cluster centers from KMeans
cluster_centers = kmeans.cluster_centers_
ordered_centers = [cluster_centers[i][0] for i in cluster_order]

# Add the cluster centers to the summary
uhi_cluster_summary['Center'] = ordered_centers
print("K-means Clustering Results:")
print(uhi_cluster_summary)

# Visualization of clusters
plt.figure(figsize=(10, 6))
for cluster in range(3):
    cluster_data = df[df['uhi_class'] == cluster]
    plt.scatter(cluster_data.index, cluster_data['UHI'], label=f'Class {cluster}', alpha=0.6)
plt.title('UHI Clusters Visualization')
plt.xlabel('Index')
plt.ylabel('UHI')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Get the cluster centers from KMeans
cluster_centers = kmeans.cluster_centers_

# Sort the cluster centers based on the mean UHI value of each cluster
ordered_centers = [cluster_centers[i][0] for i in cluster_order]

# Print the cluster centers with their corresponding class
print("Cluster Centers (Ordered by Class):")
for idx, center in enumerate(ordered_centers):
    print(f"Class {idx}: Center = {center:.2f}")


# Define features (X) and target (y)
X = df[['NDBI', 'NDVI', 'SMI', 'NO2', 'O3', 'LAI', 'PM10', 'SAVI', 'PM2.5',
         'temp_station', 'mrt', 'WDSP', 'd_or_n', 'urban_max']]
y = df['uhi_class']

# Normalize features using Min-Max scaling
X = (X - X.min()) / (X.max() - X.min())

# Split data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000),
    'Decision Tree': DecisionTreeClassifier(max_depth=5),
    'Support Vector Classifier': SVC(C=10, kernel='linear', probability=True),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'XGBoost': xgb.XGBClassifier(objective='multi:softmax', learning_rate=0.1, n_estimators=100, max_depth=5),
    'MLP Classifier': MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=5000, random_state=42)
}

# Train and evaluate each model
results = []
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(x_train, y_train)

    # Predict on test set
    preds = model.predict(x_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, preds)
    print(f"{model_name} -> Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

    # Confusion Matrix
    y_pred = model.predict(x_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Get unique labels for proper plotting
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=unique_labels)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    # Store results
    results.append({
        'Model': model_name,
        'Accuracy': accuracy
    })

# Convert results to a DataFrame for better visualization
results_df = pd.DataFrame(results)

# Display results
print("\nModel Evaluation Results:")
print(results_df)

# Save results to a CSV file
results_df.to_csv('classification_model_results.csv', index=False)
print("\nClassification results have been saved to 'classification_model_results.csv'.")

# Logistic Regression - Σημαντικότητα χαρακτηριστικών
log_reg_model = models['Logistic Regression']

# Coefficients for Logistic Regression
coefficients = log_reg_model.coef_[0]
log_reg_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(coefficients)
}).sort_values(by='Importance', ascending=False)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(log_reg_importance['Feature'], log_reg_importance['Importance'], color='lightcoral')
plt.gca().invert_yaxis()
plt.title('Σημαντικότητα Χαρακτηριστικών - Logistic Regression')
plt.xlabel('Σημαντικότητα (Απόλυτη Τιμή Συντελεστών)')
plt.show()

print("Logistic Regression Feature Importance:")
print(log_reg_importance)

rf_model = models['Random Forest']

# Random Forest feature importance
rf_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(rf_importance['Feature'], rf_importance['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Σημαντικότητα Χαρακτηριστικών - Random Forest')
plt.xlabel('Σημαντικότητα')
plt.show()

print("Random Forest Feature Importance:")
print(rf_importance)

xgb_model = models['XGBoost']

# XGBoost feature importance
xgb_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(xgb_importance['Feature'], xgb_importance['Importance'], color='lightgreen')
plt.gca().invert_yaxis()
plt.title('Σημαντικότητα Χαρακτηριστικών - XGBoost')
plt.xlabel('Σημαντικότητα')
plt.show()

print("XGBoost Feature Importance:")
print(xgb_importance)

dt_model = models['Decision Tree']

# Decision Tree feature importance
dt_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Visualization
plt.figure(figsize=(10, 6))
plt.barh(dt_importance['Feature'], dt_importance['Importance'], color='orange')
plt.gca().invert_yaxis()
plt.title('Σημαντικότητα Χαρακτηριστικών - Decision Tree')
plt.xlabel('Σημαντικότητα')
plt.show()

print("Decision Tree Feature Importance:")
print(dt_importance)

import shap

svc_model = models['Support Vector Classifier']

# Creating an explainer for SVC
explainer = shap.KernelExplainer(svc_model.predict, x_train)
shap_values_svc = explainer.shap_values(x_test, nsamples=100)

# SHAP Summary Plot
shap.summary_plot(shap_values_svc, x_test, feature_names=X.columns)

mlp_model = models['MLP Classifier']

# Creating an explainer for MLP
explainer = shap.KernelExplainer(mlp_model.predict, x_train)
shap_values_mlp = explainer.shap_values(x_test, nsamples=100)

# SHAP Summary Plot
#shap.summary_plot(shap_values_mlp, x_test, feature_names=X.columns)

# Comparative table of importance
feature_importances_combined = pd.DataFrame({
    'Feature': X.columns,
    'Logistic Regression': log_reg_importance.set_index('Feature')['Importance'],
    'Random Forest': rf_importance.set_index('Feature')['Importance'],
    'XGBoost': xgb_importance.set_index('Feature')['Importance'],
    'Decision Tree': dt_importance.set_index('Feature')['Importance']
})

# Display of table
print("Συγκριτικός Πίνακας Σημαντικότητας:")
print(feature_importances_combined)

# Visualization with heatmap 
import seaborn as sns
plt.figure(figsize=(12, 8))
sns.heatmap(feature_importances_combined.set_index('Feature'), annot=True, cmap='coolwarm')
plt.title('Συγκριτική Σημαντικότητα Χαρακτηριστικών')
plt.show()

