import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve
)
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Φόρτωμα των δεδομένων
file_path = '/Users/katerinaargyriparoni/Downloads/results_summary_2024_athens_interpolated.csv'
df_inference = pd.read_csv(file_path, delimiter=';')

# Κανονικοποίηση των ονομάτων των στηλών
df_inference.columns = df_inference.columns.str.strip()

# Επιλογή χαρακτηριστικών για πρόβλεψη
required_features = ['NDBI', 'NDVI', 'SMI', 'NO2', 'O3', 'LAI', 'PM10', 'SAVI', 'PM2.5',
                     'temp_station', 'mrt', 'WDSP', 'd_or_n', 'urban_max']

# Έλεγχος για τυχόν ελλείποντα χαρακτηριστικά
missing_features = [feat for feat in required_features if feat not in df_inference.columns]
if missing_features:
    raise ValueError(f"The following required features are missing from the dataset: {missing_features}")

# Επιλογή μόνο των απαιτούμενων χαρακτηριστικών για την πρόβλεψη
X_inference = df_inference[required_features]

# Κανονικοποίηση των χαρακτηριστικών (min-max scaling)
scaler = MinMaxScaler()
X_inference_normalized = pd.DataFrame(scaler.fit_transform(X_inference), columns=required_features)

# Κατηγοριοποίηση του UHI με KMeans clustering
uhi_values = df_inference['UHI'].values.reshape(-1, 1)
kmeans = KMeans(n_clusters=3, random_state=42)
df_inference['uhi_class'] = kmeans.fit_predict(uhi_values)

# Ταξινόμηση των clusters με βάση την μέση τιμή του UHI
cluster_order = df_inference.groupby('uhi_class')['UHI'].mean().sort_values().index
cluster_mapping = {old: new for new, old in enumerate(cluster_order)}

# Αντιστοίχιση των clusters σε ταξινομημένες κατηγορίες
df_inference['uhi_class'] = df_inference['uhi_class'].map(cluster_mapping)

# Εκτύπωση των πρώτων γραμμών για να δούμε τη στήλη 'uhi_class'
print(df_inference[['UHI', 'uhi_class']].head())

# Φόρτωση των αποθηκευμένων μοντέλων
model_paths = [
    'logistic_regression_model.joblib',
    'decision_tree_model.joblib',
    'support_vector_classifier_model.joblib',
    'random_forest_model.joblib',
    'xgboost_model.joblib',
    'mlp_classifier_model.joblib'
]

# Πραγματικές ετικέτες (target) για το inference
y_true = df_inference['uhi_class']

# Αποθήκευση των αποτελεσμάτων για κάθε μοντέλο
model_results = {}

for model_path in model_paths:
    # Φόρτωση του μοντέλου
    model_name = model_path.split('_model.joblib')[0].replace('_', ' ').title()
    loaded_model = joblib.load(model_path)
    print(f"Model loaded: {model_name} from {model_path}")

    # Πρόβλεψη των κλάσεων
    y_pred = loaded_model.predict(X_inference_normalized)
    y_prob = loaded_model.predict_proba(X_inference_normalized)  # Για υπολογισμό των πιθανοτήτων

    # Προσθήκη προβλέψεων στο dataframe
    df_inference[f'{model_name}_prediction'] = y_pred

    # Υπολογισμός των metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    logloss = log_loss(y_true, y_prob)

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(y_true))

    # Οπτικοποίηση Confusion Matrix
    plt.figure(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    # Οπτικοποίηση ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1], pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.show()

    # Αποθήκευση αποτελεσμάτων
    model_results[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC Score': roc_auc,
        'Log Loss': logloss
    }

    # Εκτύπωση classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred, zero_division=0))

# Μετατροπή των αποτελεσμάτων σε DataFrame
results_df = pd.DataFrame.from_dict(model_results, orient='index')

# Εμφάνιση αποτελεσμάτων
print("\nModel Evaluation Results:")
print(results_df)

# Αποθήκευση των αποτελεσμάτων σε αρχείο CSV
results_df.to_csv('inference_model_evaluation_metrics.csv', index=True)
print("\nModel evaluation results have been saved to 'inference_model_evaluation_metrics.csv'.")

# Αποθήκευση του dataframe με τις προβλέψεις
output_path = 'inference_results_with_all_predictions.csv'
df_inference.to_csv(output_path, index=False)
print(f"\nInference completed. Results saved to: {output_path}")