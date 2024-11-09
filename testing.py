# -*- coding: utf-8 -*-
"""Optimized_EEG_Suicide_Risk_Prediction.ipynb

Optimized for predicting suicide risk categories for individuals using EEG data.
"""

# -----------------------------------
# 0. Setup and Import Libraries
# -----------------------------------

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input,
    Attention, GlobalAveragePooling1D,
    Bidirectional, GRU, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# For handling imbalanced data
from imblearn.over_sampling import SMOTE
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from imblearn.metrics import classification_report_imbalanced

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Add these imports
import os
from IPython.display import display

# Create plots directory
os.makedirs('plots', exist_ok=True)

# Function to safely display in both notebook and script environments
def safe_display(df):
    try:
        display(df)
    except:
        print(df)

# Function to save plot to plots directory and close it
def save_plot(plt, filename):
    """Save plot to plots directory and close it"""
    plt.savefig(os.path.join('plots', filename))
    plt.close()

# -----------------------------------
# 1. Load and Inspect the Dataset
# -----------------------------------

# Define the file path
file_path = 'EEG.machinelearing_data_BRMH.csv'  # Update the path as needed

# Load the dataset
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    raise FileNotFoundError(f"The file at {file_path} was not found. Please check the path and try again.")

# Display the first few rows
print("\nFirst 5 rows of the dataset:")
safe_display(data.head())

# Check for missing values
print("\nMissing values per column:")
safe_display(data.isnull().sum())

# -----------------------------------
# 2. Define Suicide Rate Mapping and Assign Labels
# -----------------------------------

# Define the suicide rate mapping
suicide_rate_dict = {
    'Addictive disorder': 0.025,
    'Trauma and stress-related disorder': 0.025,
    'Mood disorder': 0.10,
    'Healthy control': 0.0001,
    'Obsessive-compulsive disorder': 0.005,
    'Schizophrenia': 0.05,
    'Anxiety disorder': 0.0075
}

# Map 'main.disorder' to 'suicide_rate'
data['suicide_rate'] = data['main.disorder'].map(suicide_rate_dict)

# Handle missing suicide rates
missing_rates = data['suicide_rate'].isnull().sum()
if missing_rates > 0:
    print(f"\nNumber of entries with missing suicide rates: {missing_rates}")
    mean_suicide_rate = data['suicide_rate'].mean()
    data['suicide_rate'] = data['suicide_rate'].fillna(mean_suicide_rate)
    print(f"Filled missing suicide rates with the mean value: {mean_suicide_rate:.4f}")

# Verify the mapping
print("\nSuicide rate distribution by main disorder:")
safe_display(data.groupby('main.disorder')['suicide_rate'].mean())

# Assign suicide labels based on suicide_rate probabilities
def assign_suicide_label(row):
    # Generate a random number between 0 and 1
    rand_num = np.random.uniform(0, 1)
    return 1 if rand_num < row['suicide_rate'] else 0

# Apply the function to assign labels
data['suicide_label'] = data.apply(assign_suicide_label, axis=1)

# Check the distribution of labels
print("\nSuicide Label Distribution:")
safe_display(data['suicide_label'].value_counts())

# -----------------------------------
# 3. Load Feature Metadata
# -----------------------------------

# EEG Feature Metadata as CSV-formatted string
csv_metadata = """Feature,Healthy_Mean_Energy_Level,Non_Healthy_Mean_Energy_Level,Difference_Energy_Level,Brain_Region,Lateralization,Implications
COH.A.delta.m.T5,65.958037,52.486650,13.471388,Temporal Lobe,Left,"Associated with auditory processing and language comprehension."
COH.B.theta.m.T5.n.P3,67.056235,55.822028,11.234207,Temporal Lobe,Left,"Associated with auditory processing and language comprehension."
COH.A.delta.a.FP1.e.Fz,70.068086,59.417262,10.650824,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.p.P4.q.T6,67.217933,56.827849,10.390084,Temporal Lobe,Right,"Linked to emotional processing and memory encoding."
COH.A.delta.m.T5.r.O1,58.495726,48.833212,9.662515,Occipital Lobe,Left,"Tied to visual processing and vigilance."
COH.A.delta.a.FP1.f.F4,62.605702,53.486447,9.119255,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.a.FP1.b.FP2,76.076310,67.146066,8.930244,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.B.theta.a.FP1.e.Fz,79.368891,71.541124,7.827766,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.k.C4.q.T6,49.916102,42.248086,7.668016,Central-Frontal Lobe,Right,"Associated with attention, cognitive control, and motor functions."
COH.A.delta.c.F7.e.Fz,47.729256,40.080630,7.648626,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.f.F4.g.F8,68.213670,60.652938,7.560733,Frontal Lobe,Right,"Linked to emotional regulation and attentional focus."
COH.C.alpha.n.P3.p.P4,43.896760,51.314581,7.417820,Parietal Lobe,Left,"Related to sensory integration and spatial awareness."
COH.A.delta.d.F3.e.Fz,81.161198,74.017626,7.143572,Central-Frontal Lobe,Left,"Involved in executive function, working memory, and motor planning."
COH.E.highbeta.a.FP1.e.Fz,71.102721,64.040806,7.061915,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
AB.C.alpha.s.O2,20.192476,27.180825,6.988350,Occipital Lobe,Right,"Involved in visual processing and sensory integration."
COH.E.highbeta.a.FP1.b.FP2,72.811988,65.825375,6.986613,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.B.theta.a.FP1.f.F4,72.471040,65.509472,6.961569,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.b.FP2.f.F4,69.024622,62.106630,6.917993,Frontal Lobe,Right,"Linked to emotional regulation and attentional focus."
COH.A.delta.b.FP2.c.F7,40.394434,33.496576,6.897857,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.C.alpha.k.C4.n.P3,34.202367,41.089249,6.886881,Central-Frontal Lobe,Right,"Associated with attention, cognitive control, and motor functions."
COH.E.highbeta.m.T5.n.P3,68.746103,61.884273,6.861829,Temporal Lobe,Left,"Associated with auditory processing and language comprehension."
COH.A.delta.o.Pz.q.T6,50.108448,43.289470,6.818978,Temporal Lobe,Right,"Linked to emotional processing and memory encoding."
COH.D.beta.a.FP1.e.Fz,79.502451,72.686871,6.815580,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.a.FP1.j.Cz,44.951555,38.153357,6.798198,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.E.highbeta.a.FP1.d.F3,75.014311,68.346177,6.668133,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.F.gamma.m.T5.n.P3,70.390711,63.728167,6.662544,Temporal Lobe,Left,"Associated with auditory processing and language comprehension."
COH.C.alpha.i.C3.p.P4,33.263954,39.894513,6.630558,Central-Frontal Lobe,Left,"Involved in executive function, working memory, and motor planning."
COH.D.beta.b.FP2.d.F3,68.921708,62.357394,6.564314,Frontal Lobe,Right,"Linked to emotional regulation and attentional focus."
AB.C.alpha.r.O1,20.706044,27.242311,6.536268,Occipital Lobe,Left,"Tied to visual processing and vigilance."
COH.F.gamma.a.FP1.e.Fz,70.951554,64.433588,6.517966,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.B.theta.m.T5.o.Pz,40.719662,34.216074,6.503588,Temporal Lobe,Left,"Associated with auditory processing and language comprehension."
COH.B.theta.b.FP2.e.Fz,77.527766,71.040121,6.487646,Frontal Lobe,Right,"Linked to emotional regulation and attentional focus."
COH.A.delta.f.F4.k.C4,71.940748,65.519208,6.421541,Central-Frontal Lobe,Right,"Associated with attention, cognitive control, and motor functions."
COH.A.delta.e.Fz.i.C3,67.497395,61.097446,6.399949,Central-Frontal Lobe,Left,"Involved in executive function, working memory, and motor planning."
COH.F.gamma.a.FP1.b.FP2,72.052558,65.656916,6.395643,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.n.P3.r.O1,57.590783,51.254106,6.336677,Occipital Lobe,Left,"Tied to visual processing and vigilance."
COH.A.delta.a.FP1.i.C3,47.936441,41.650618,6.285823,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.C.alpha.j.Cz.p.P4,45.357392,51.619384,6.261992,Parietal Lobe,Right,"Involved in attention and sensory integration."
COH.D.beta.p.P4.q.T6,69.071030,62.843271,6.227760,Temporal Lobe,Right,"Linked to emotional processing and memory encoding."
COH.A.delta.j.Cz.m.T5,35.284470,29.057678,6.226793,Temporal Lobe,Left,"Associated with auditory processing and language comprehension."
COH.B.theta.b.FP2.d.F3,69.961286,63.741118,6.220168,Frontal Lobe,Right,"Linked to emotional regulation and attentional focus."
COH.A.delta.c.F7.i.C3,49.252274,43.056524,6.195750,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.B.theta.p.P4.s.O2,63.165701,57.000417,6.165285,Occipital Lobe,Right,"Involved in visual processing and sensory integration."
COH.A.delta.r.O1.s.O2,56.442676,50.280871,6.161805,Occipital Lobe,Left,"Tied to visual processing and vigilance."
COH.A.delta.c.F7.h.T3,60.055023,53.909612,6.145411,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.B.theta.c.F7.e.Fz,57.367251,51.233342,6.133909,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.C.alpha.e.Fz.n.P3,25.036765,31.088452,6.051687,Parietal Lobe,Left,"Related to sensory integration and spatial awareness."
COH.A.delta.i.C3.m.T5,45.147996,39.116258,6.031738,Central-Frontal Lobe,Left,"Involved in executive function, working memory, and motor planning."
AB.C.alpha.p.P4,27.253130,33.274890,6.021760,Parietal Lobe,Right,"Involved in attention and sensory integration."
COH.D.beta.a.FP1.d.F3,81.167903,75.146580,6.021323,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.C.alpha.d.F3.n.P3,27.533992,33.551879,6.017887,Central-Frontal Lobe,Left,"Involved in executive function, working memory, and motor planning."
COH.F.gamma.a.FP1.d.F3,73.616400,67.633651,5.982749,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.A.delta.a.FP1.k.C4,39.776036,33.793306,5.982730,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.B.theta.a.FP1.b.FP2,83.343328,77.414850,5.928478,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
COH.C.alpha.e.Fz.p.P4,26.373669,32.252634,5.878965,Parietal Lobe,Right,"Involved in attention and sensory integration."
COH.A.delta.d.F3.f.F4,69.548083,63.714451,5.833633,Central-Frontal Lobe,Left,"Involved in executive function, working memory, and motor planning."
COH.D.beta.a.FP1.b.FP2,81.332333,75.539912,5.792421,Frontal Lobe,Left,"Associated with impulse control, attention regulation, and emotional processing."
"""

# Read the metadata into a DataFrame
try:
    feature_metadata = pd.read_csv(StringIO(csv_metadata))
    print("\nFeature Metadata Loaded Successfully:")
    safe_display(feature_metadata.head())
except pd.errors.ParserError as e:
    print(f"\nParserError: {e}")

# -----------------------------------
# 4. Data Preprocessing
# -----------------------------------

def preprocess_data(data, feature_metadata, non_feature_columns, target_column='suicide_label'):
    """
    Preprocess the data for binary classification.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - feature_metadata: pandas DataFrame containing feature metadata.
    - non_feature_columns: list of column names to exclude from features.
    - target_column: string, name of the target column.

    Returns:
    - X_scaled_df: pandas DataFrame of preprocessed features.
    - y: numpy array of target labels.
    - scaler: fitted StandardScaler object.
    - imputer: fitted SimpleImputer object.
    - feature_names_extended: list of feature names after preprocessing.
    """
    # Extract feature names from metadata
    top_features = feature_metadata['Feature'].tolist()

    # Check available features in the dataset
    available_features = [feature for feature in top_features if feature in data.columns]
    missing_features = [feature for feature in top_features if feature not in data.columns]

    if missing_features:
        print(f"\nMissing features that are not in the dataset and will be excluded: {missing_features}")

    print(f"\nNumber of available features for modeling: {len(available_features)}")

    # Select features
    X = data[available_features]

    # **Feature Engineering: Adding Statistical Features**
    X['EEG_mean'] = X.mean(axis=1)
    X['EEG_std'] = X.std(axis=1)

    # Update feature names
    feature_names_extended = available_features + ['EEG_mean', 'EEG_std']

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=feature_names_extended)

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names_extended)

    # Define target variable
    y = data[target_column].values

    return X_scaled_df, y, scaler, imputer, feature_names_extended

# Define non-feature columns
non_feature_columns = ['no.', 'sex', 'age', 'eeg.date', 'education', 'IQ', 'main.disorder', 'specific.disorder']

# Preprocess the data
X, y, scaler, imputer, feature_names = preprocess_data(data, feature_metadata, non_feature_columns)

# -----------------------------------
# 5. Handle Class Imbalance with SMOTE
# -----------------------------------

print("\nHandling class imbalance with SMOTE...")

smote = SMOTE(random_state=SEED)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("After SMOTE, class distribution:")
print(pd.Series(y_resampled).value_counts())

# -----------------------------------
# 6. Train-Test Split
# -----------------------------------

def split_data(X, y, test_size=0.2, random_state=SEED):
    """
    Split the data into training and testing sets with stratification.

    Parameters:
    - X: pandas DataFrame of features.
    - y: numpy array of target labels.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, random seed.

    Returns:
    - X_train, X_test, y_train, y_test: split datasets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

X_train, X_test, y_train, y_test = split_data(X_resampled, y_resampled)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# -----------------------------------
# 7. Model Implementations
# -----------------------------------

# 7.1 Enhanced LSTM Model with Attention and Bidirectional Layers
def build_enhanced_lstm_model(input_shape):
    """
    Build an Enhanced LSTM model with Attention and Bidirectional layers for binary classification.

    Parameters:
    - input_shape: tuple, shape of the input data (time_steps, features).

    Returns:
    - model: compiled Keras model.
    """
    inputs = Input(shape=input_shape)
    # Adding a Bidirectional LSTM layer
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, activation='tanh'))(inputs)
    dropout_out = Dropout(0.3)(lstm_out)
    # Attention Mechanism
    attention_out = Attention()([dropout_out, dropout_out])
    context_vector = GlobalAveragePooling1D()(attention_out)
    dense_out = Dense(128, activation='relu')(context_vector)
    dropout_dense = Dropout(0.3)(dense_out)
    output = Dense(1, activation='sigmoid')(dropout_dense)
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 7.2 Random Forest Classifier
def build_random_forest():
    """
    Build a Random Forest classifier with balanced class weights.

    Returns:
    - rf: RandomForestClassifier instance.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
    return rf

# 7.3 XGBoost Classifier
def build_xgboost():
    """
    Build an XGBoost classifier configured for binary classification.

    Returns:
    - xgbc: XGBClassifier instance.
    """
    xgbc = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=1  # Adjust based on class imbalance if needed
    )
    return xgbc

# -----------------------------------
# 8. Hyperparameter Tuning
# -----------------------------------

def tune_random_forest(X, y):
    """
    Perform hyperparameter tuning for Random Forest using RandomizedSearchCV.

    Parameters:
    - X: pandas DataFrame of features.
    - y: numpy array of target labels.

    Returns:
    - best_rf: RandomForestClassifier with the best found parameters.
    """
    rf = RandomForestClassifier(random_state=SEED, class_weight='balanced')
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    randomized_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='roc_auc',  # ROC AUC for binary classification
        random_state=SEED,
        n_jobs=-1,
        verbose=1
    )
    randomized_search.fit(X, y)
    print(f"Best parameters for Random Forest: {randomized_search.best_params_}")
    best_rf = randomized_search.best_estimator_
    return best_rf

def tune_xgboost(X, y):
    """
    Perform hyperparameter tuning for XGBoost using RandomizedSearchCV.

    Parameters:
    - X: pandas DataFrame of features.
    - y: numpy array of target labels.

    Returns:
    - best_xgbc: XGBClassifier with the best found parameters.
    """
    xgbc = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=SEED,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    randomized_search = RandomizedSearchCV(
        estimator=xgbc,
        param_distributions=param_distributions,
        n_iter=50,
        cv=5,
        scoring='roc_auc',  # ROC AUC for binary classification
        random_state=SEED,
        n_jobs=-1,
        verbose=1
    )
    randomized_search.fit(X, y)
    print(f"Best parameters for XGBoost: {randomized_search.best_params_}")
    best_xgbc = randomized_search.best_estimator_
    return best_xgbc

# -----------------------------------
# 9. Train Models
# -----------------------------------

# 9.1 Train Enhanced LSTM Model
def train_lstm(X_train, y_train, X_val, y_val, input_shape):
    """
    Train the Enhanced LSTM model with EarlyStopping.

    Parameters:
    - X_train: numpy array of training features reshaped for LSTM.
    - y_train: numpy array of training labels.
    - X_val: numpy array of validation features reshaped for LSTM.
    - y_val: numpy array of validation labels.
    - input_shape: tuple, shape of the input data (time_steps, features).

    Returns:
    - model: trained Keras model.
    - history: training history.
    """
    model = tf.keras.Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Bidirectional(LSTM(32)),
        Dense(16, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

# 9.2 Train Random Forest
def train_random_forest_model(X_train, y_train):
    """
    Train the Random Forest classifier.

    Parameters:
    - X_train: pandas DataFrame of training features.
    - y_train: numpy array of training labels.

    Returns:
    - rf: trained RandomForestClassifier.
    """
    rf = build_random_forest()
    rf.fit(X_train, y_train)
    return rf

# 9.3 Train XGBoost
def train_xgboost_model(X_train, y_train):
    """
    Train the XGBoost classifier.

    Parameters:
    - X_train: pandas DataFrame of training features.
    - y_train: numpy array of training labels.

    Returns:
    - xgbc: trained XGBClassifier.
    """
    xgbc = build_xgboost()
    xgbc.fit(X_train, y_train)
    return xgbc

# -----------------------------------
# 10. Model Evaluation
# -----------------------------------

def evaluate_model_classification(y_true, y_pred, y_pred_proba, model_name):
    """
    Evaluate the model using various metrics and display the classification report.

    Parameters:
    - y_true: numpy array of true labels.
    - y_pred: numpy array of predicted labels.
    - y_pred_proba: numpy array of predicted probabilities.
    - model_name: string, name of the model.

    Returns:
    - None
    """
    print(f"\n{model_name} Performance Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix_cm(y_true, y_pred, model_name):
    """
    Plot the confusion matrix for binary classification.

    Parameters:
    - y_true: numpy array of true labels.
    - y_pred: numpy array of predicted labels.
    - model_name: string, name of the model.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_plot(plt, f'confusion_matrix_{model_name.replace(" ", "_")}.png')

def plot_roc_curve_custom(y_true, y_pred_proba, model_name):
    """
    Plot ROC curve for binary classification.

    Parameters:
    - y_true: numpy array of true labels.
    - y_pred_proba: numpy array of predicted probabilities.
    - model_name: string, name of the model.

    Returns:
    - None
    """
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    save_plot(plt, f'roc_curve_{model_name.replace(" ", "_")}.png')

# -----------------------------------
# 11. Feature Importance Analysis
# -----------------------------------

def plot_feature_importance(model, feature_names, model_type):
    """
    Plot feature importance for tree-based models.

    Parameters:
    - model: trained model instance.
    - feature_names: list of feature names.
    - model_type: string, type of the model ("RandomForest", "XGBoost").

    Returns:
    - None
    """
    if model_type in ["RandomForest", "XGBoost"]:
        plt.figure(figsize=(10, 6))
        importance = model.feature_importances_
        indices = np.argsort(importance)[-10:]
        plt.barh(range(10), importance[indices])
        plt.yticks(range(10), [feature_names[i] for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(f'Top 10 Important Features - {model_type}')
        plt.tight_layout()
        save_plot(plt, f'feature_importance_{model_type}.png')

# -----------------------------------
# 12. SHAP Analysis
# -----------------------------------

def shap_analysis_classification(model, X_test, feature_names, model_type):
    """
    Perform SHAP analysis for binary classification models.
    """
    print("\nPerforming SHAP Analysis...")
    if model_type in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Convert feature_names to numpy array if it's not already
        feature_names = np.array(feature_names)
        
        # Summary plot - handle both single and multi-output cases
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            # For multi-output models
            shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
        else:
            # For single output models
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        save_plot(plt, f'shap_summary_{model_type}.png')
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        if isinstance(shap_values, list):
            shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar", show=False)
        save_plot(plt, f'shap_bar_{model_type}.png')
    
    elif model_type == "LSTM":
        print("SHAP analysis for LSTM models is not implemented in this version")

# -----------------------------------
# 13. Save and Load Models
# -----------------------------------

def save_model_custom(model, filename, model_type="RandomForest"):
    """
    Save the trained model to disk.

    Parameters:
    - model: trained model instance.
    - filename: string, path to save the model.
    - model_type: string, type of the model ("RandomForest", "XGBoost", "LSTM").

    Returns:
    - None
    """
    if model_type == "LSTM":
        model.save(filename)
    else:
        import joblib
        joblib.dump(model, filename)
    print(f"Model saved to '{filename}'")

def load_model_custom(filename, model_type="RandomForest"):
    """
    Load the trained model from disk.

    Parameters:
    - filename: string, path to load the model from.
    - model_type: string, type of the model ("RandomForest", "XGBoost", "LSTM").

    Returns:
    - model: loaded model instance.
    """
    if model_type == "LSTM":
        model = load_model(filename)
    else:
        import joblib
        model = joblib.load(filename)
    print(f"Model loaded from '{filename}'")
    return model

# -----------------------------------
# 14. Make Predictions on New Samples
# -----------------------------------

def make_new_prediction_classification(model, new_sample, scaler, imputer, feature_names, model_type):
    """
    Make a prediction for a new sample.
    """
    try:
        # Convert to pandas DataFrame
        new_sample_df = pd.DataFrame([new_sample], columns=feature_names)

        # Handle missing values
        new_sample_imputed = imputer.transform(new_sample_df)

        # Scale the features
        new_sample_scaled = scaler.transform(new_sample_imputed)

        if model_type == "LSTM":
            # Reshape for LSTM
            new_sample_reshaped = new_sample_scaled.reshape((1, 1, new_sample_scaled.shape[1]))
            prediction_proba = model.predict(new_sample_reshaped).flatten()[0]
            prediction = 1 if prediction_proba > 0.5 else 0
        else:
            prediction_proba = model.predict_proba(new_sample_scaled)[0][1]  # Get probability for class 1
            prediction = model.predict(new_sample_scaled)[0]

        print(f"\nPredicted Suicide Risk for the new sample:")
        print(f"Probability of Suicide: {prediction_proba:.6f}")
        print(f"Predicted Label: {'Suicide' if prediction == 1 else 'No Suicide'}")
        
        return prediction, prediction_proba
        
    except Exception as e:
        print(f"Error making prediction: {str(e)}")
        return None, None

# -----------------------------------
# 15. Main Execution Flow
# -----------------------------------

# Further split training data for validation (for LSTM)
X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
    X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
)

# Reshape data for LSTM [samples, time steps, features]
X_train_lstm_reshaped = X_train_lstm.values.reshape((X_train_lstm.shape[0], 1, X_train_lstm.shape[1]))
X_val_lstm_reshaped = X_val_lstm.values.reshape((X_val_lstm.shape[0], 1, X_val_lstm.shape[1]))
X_test_lstm_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Train Enhanced LSTM Model
print("\nTraining Enhanced LSTM Model with Attention and Bidirectional Layers...")
enhanced_lstm_model, enhanced_lstm_history = train_lstm(
    X_train_lstm_reshaped, y_train_lstm,
    X_val_lstm_reshaped, y_val_lstm,
    (1, X_train_lstm.shape[1])
)

# Plot Enhanced LSTM Training History
plt.figure(figsize=(10,6))
plt.plot(enhanced_lstm_history.history['loss'], label='Train Loss')
plt.plot(enhanced_lstm_history.history['val_loss'], label='Validation Loss')
plt.title('Enhanced LSTM Model Loss Over Epochs')
plt.ylabel('Loss (Binary Crossentropy)')
plt.xlabel('Epoch')
plt.legend()
plt.savefig('plots/lstm_loss.png')
plt.close()

# Evaluate Enhanced LSTM Model
print("\nEvaluating Enhanced LSTM Model on Test Set...")
enhanced_lstm_test_proba = enhanced_lstm_model.predict(X_test_lstm_reshaped).flatten()
enhanced_lstm_test_pred = (enhanced_lstm_test_proba > 0.5).astype(int)
evaluate_model_classification(y_test, enhanced_lstm_test_pred, enhanced_lstm_test_proba, "Enhanced LSTM")
plot_confusion_matrix_cm(y_test, enhanced_lstm_test_pred, "Enhanced LSTM")
plot_roc_curve_custom(y_test, enhanced_lstm_test_proba, "Enhanced LSTM")

# Train and Tune Random Forest
print("\nTraining and Tuning Random Forest Classifier...")
best_rf = tune_random_forest(X_train, y_train)
rf_test_pred = best_rf.predict(X_test)
rf_test_proba = best_rf.predict_proba(X_test)[:,1]
evaluate_model_classification(y_test, rf_test_pred, rf_test_proba, "Random Forest")
plot_confusion_matrix_cm(y_test, rf_test_pred, "Random Forest")
plot_roc_curve_custom(y_test, rf_test_proba, "Random Forest")
plot_feature_importance(best_rf, feature_names, "RandomForest")

# Train and Tune XGBoost
print("\nTraining and Tuning XGBoost Classifier...")
best_xgbc = tune_xgboost(X_train, y_train)
xgbc_test_pred = best_xgbc.predict(X_test)
xgbc_test_proba = best_xgbc.predict_proba(X_test)[:,1]
evaluate_model_classification(y_test, xgbc_test_pred, xgbc_test_proba, "XGBoost")
plot_confusion_matrix_cm(y_test, xgbc_test_pred, "XGBoost")
plot_roc_curve_custom(y_test, xgbc_test_proba, "XGBoost")
plot_feature_importance(best_xgbc, feature_names, "XGBoost")

# Compare Models
comparison_df = pd.DataFrame({
    'Model': ['Enhanced LSTM', 'Random Forest', 'XGBoost'],
    'Accuracy': [
        accuracy_score(y_test, enhanced_lstm_test_pred),
        accuracy_score(y_test, rf_test_pred),
        accuracy_score(y_test, xgbc_test_pred)
    ],
    'Precision': [
        precision_score(y_test, enhanced_lstm_test_pred, zero_division=0),
        precision_score(y_test, rf_test_pred, zero_division=0),
        precision_score(y_test, xgbc_test_pred, zero_division=0)
    ],
    'Recall': [
        recall_score(y_test, enhanced_lstm_test_pred, zero_division=0),
        recall_score(y_test, rf_test_pred, zero_division=0),
        recall_score(y_test, xgbc_test_pred, zero_division=0)
    ],
    'F1_Score': [
        f1_score(y_test, enhanced_lstm_test_pred, zero_division=0),
        f1_score(y_test, rf_test_pred, zero_division=0),
        f1_score(y_test, xgbc_test_pred, zero_division=0)
    ],
    'ROC_AUC': [
        roc_auc_score(y_test, enhanced_lstm_test_proba),
        roc_auc_score(y_test, rf_test_proba),
        roc_auc_score(y_test, xgbc_test_proba)
    ]
})

print("\nModel Comparison:")
print(comparison_df.sort_values(by='ROC_AUC', ascending=False).to_string(index=False))

# Select the best model based on ROC AUC Score
best_model_row = comparison_df.sort_values(by='ROC_AUC', ascending=False).iloc[0]
best_model_name = best_model_row['Model']
print(f"\nBest Model Based on ROC AUC Score: {best_model_name}")

# -----------------------------------
# 16. SHAP Analysis on the Best Model
# -----------------------------------

if best_model_name == "Random Forest":
    shap_model = best_rf
    model_type = "RandomForest"
elif best_model_name == "XGBoost":
    shap_model = best_xgbc
    model_type = "XGBoost"
elif best_model_name == "Enhanced LSTM":
    shap_model = enhanced_lstm_model
    model_type = "LSTM"
else:
    raise ValueError("Unsupported model selected for SHAP analysis.")

# Perform SHAP Analysis
shap_analysis_classification(shap_model, X_test, feature_names, model_type)

# -----------------------------------
# 17. Save the Best Model
# -----------------------------------

model_filename = f'best_model_{best_model_name.replace(" ", "_")}.joblib' if best_model_name in ["Random Forest", "XGBoost"] else f'best_model_{best_model_name.replace(" ", "_")}.h5'

if best_model_name in ["Random Forest", "XGBoost"]:
    save_model_custom(shap_model, model_filename, model_type=best_model_name)
elif best_model_name == "Enhanced LSTM":
    save_model_custom(shap_model, model_filename, model_type=best_model_name)

# -----------------------------------
# 18. Load the Best Model and Make Predictions
# -----------------------------------

model_filename = f'best_model_{best_model_name.replace(" ", "_")}.joblib' if best_model_name in ["Random Forest", "XGBoost"] else f'best_model_{best_model_name.replace(" ", "_")}.h5'

# Save the best model
save_model_custom(shap_model, model_filename, model_type=best_model_name)

# Load the best model for predictions
loaded_model = load_model_custom(model_filename, model_type=best_model_name)

# -----------------------------------
# 19. Making Predictions on New Samples
# -----------------------------------

# Example: Predicting on a new sample
original_new_sample = X_test.iloc[0].values
# Pass the actual model object instead of the model name
make_new_prediction_classification(loaded_model, original_new_sample, scaler, imputer, feature_names, best_model_name)

# -----------------------------------
# 20. Conclusion
# -----------------------------------

print("\nOptimization and model improvement completed. Review the evaluation metrics and SHAP analysis to understand model performance and feature importance.")

# Add these functions before the main execution flow
def validate_model_performance(X, y, best_model, model_name="Random Forest"):
    """
    Comprehensive model validation including k-fold CV, calibration check,
    and imbalanced classification metrics.
    """
    print(f"\n{'='*20} Model Validation {'='*20}")
    
    # 1. K-fold Cross Validation
    print("\n1. Performing 5-fold Cross Validation...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_model, X, y, cv=kf, scoring='roc_auc')
    print(f"Cross-validation ROC AUC scores: {cv_scores}")
    print(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # 2. Hold-out Validation
    print("\n2. Performing Hold-out Validation...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Get detailed metrics for imbalanced classification
    print("\nImbalanced Classification Report:")
    y_pred = best_model.predict(X_val)
    print(classification_report_imbalanced(y_val, y_pred))

    # 3. Calibration Check
    print("\n3. Checking Model Calibration...")
    prob_pos = best_model.predict_proba(X_val)[:, 1]
    prob_true, prob_pred = calibration_curve(y_val, prob_pos, n_bins=10)
    
    # Calculate Brier Score
    brier = brier_score_loss(y_val, prob_pos)
    print(f"Brier Score: {brier:.4f} (lower is better)")

    # 4. Plot Calibration Curve
    plt.figure(figsize=(10, 6))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{model_name}')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    save_plot(plt, 'calibration_curve.png')

    # 5. Check Class Distribution
    print("\n4. Class Distribution in Training Data:")
    class_dist = np.bincount(y)
    print(f"Class 0 (No Suicide): {class_dist[0]}")
    print(f"Class 1 (Suicide): {class_dist[1]}")
    print(f"Class Imbalance Ratio: 1:{class_dist[0]/class_dist[1]:.2f}")

    # 6. Feature Importance Visualization (if using Random Forest)
    if hasattr(best_model, 'feature_importances_'):
        plt.figure(figsize=(12, 6))
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        sns.barplot(data=feature_importance.head(15), 
                   x='importance', y='feature')
        plt.title('Top 15 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        save_plot(plt, 'feature_importance_validation.png')

    return {
        'cv_scores': cv_scores,
        'brier_score': brier,
        'class_distribution': class_dist
    }

def calibrate_model(model, X, y):
    """
    Calibrate model probabilities using Platt Scaling
    """
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    calibrated_model = CalibratedClassifierCV(model, cv=5, method='sigmoid')
    calibrated_model.fit(X_train, y_train)
    
    # Compare original vs calibrated probabilities
    orig_prob = model.predict_proba(X_val)[:, 1]
    cal_prob = calibrated_model.predict_proba(X_val)[:, 1]
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.hist(orig_prob, bins=50, alpha=0.5, label='Original')
    plt.hist(cal_prob, bins=50, alpha=0.5, label='Calibrated')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')
    plt.title('Original vs Calibrated Probability Distribution')
    plt.legend()
    save_plot(plt, 'probability_calibration_comparison.png')
    
    return calibrated_model

# Add this after selecting the best model (after line 846)
print("\nPerforming comprehensive model validation...")
validation_results = validate_model_performance(X_resampled, y_resampled, shap_model, best_model_name)

print("\nCalibrating model probabilities...")
calibrated_model = calibrate_model(shap_model, X_resampled, y_resampled)

# Save the calibrated model
calibrated_model_filename = f'calibrated_model_{best_model_name.replace(" ", "_")}.joblib'
save_model_custom(calibrated_model, calibrated_model_filename, model_type=best_model_name)
print(f"\nCalibrated model saved as: {calibrated_model_filename}")

# Add the AdvancedNeuralArchitecture class and helper functions after your existing model classes
class AdvancedNeuralArchitecture:
    def __init__(self, input_shape, architecture_type='lstm'):
        self.input_shape = input_shape
        self.architecture_type = architecture_type
        self.models = {
            'lstm': self._create_lstm,
            'bilstm': self._create_bilstm,
            'gru': self._create_gru
        }

    def _create_lstm(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def _create_bilstm(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=self.input_shape))
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def _create_gru(self):
        model = Sequential()
        model.add(GRU(128, input_shape=self.input_shape, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(GRU(64))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def get_model(self):
        return self.models[self.architecture_type]()

def lr_schedule(epoch):
    """Learning rate scheduler with warm-up and decay"""
    initial_lr = 0.001
    if epoch < 5:  # Warm-up phase
        return float(initial_lr * ((epoch + 1) / 5))  # Explicitly cast to float
    else:  # Decay phase
        return float(initial_lr * np.exp(0.1 * (10 - epoch)))  # Use np.exp and cast to float

def prepare_sequence_data(X, sequence_length=5):
    """Prepare sequential data with overlapping windows"""
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
    return np.array(sequences)

def cross_validate_neural_models(X, y, n_splits=5):
    """Perform cross-validation with different neural architectures"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    architectures = ['lstm', 'bilstm', 'gru']
    results = {arch: {'auc': [], 'acc': []} for arch in architectures}
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        LearningRateScheduler(lr_schedule)
    ]

    sequence_length = 5
    X_seq = prepare_sequence_data(X, sequence_length)
    
    for arch in architectures:
        print(f"\nTraining {arch.upper()} model...")
        fold = 1
        
        for train_idx, val_idx in kf.split(X_seq):
            print(f"\nFold {fold}/{n_splits}")
            
            X_train, X_val = X_seq[train_idx], X_seq[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model_creator = AdvancedNeuralArchitecture(
                input_shape=(sequence_length, X.shape[1]),
                architecture_type=arch
            )
            model = model_creator.get_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            try:
                history = model.fit(
                    X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                
                y_pred = model.predict(X_val)
                results[arch]['auc'].append(roc_auc_score(y_val, y_pred))
                results[arch]['acc'].append(accuracy_score(y_val, y_pred.round()))
                
                plot_learning_curves(history, arch, fold)
                
            except Exception as e:
                print(f"Error in fold {fold} for {arch}: {str(e)}")
                continue
                
            fold += 1
    
    return results

def plot_learning_curves(history, architecture, fold):
    """Plot and save learning curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title(f'{architecture.upper()} Loss Curves - Fold {fold}')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title(f'{architecture.upper()} Accuracy Curves - Fold {fold}')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    save_plot(plt, f'learning_curves_{architecture}_fold_{fold}.png')

def print_neural_results(results):
    """Print comprehensive results summary"""
    print("\nNeural Model Performance Summary:")
    print("=" * 50)
    for arch in results:
        print(f"\n{arch.upper()} Results:")
        print(f"Mean ROC AUC: {np.mean(results[arch]['auc']):.4f} ± {np.std(results[arch]['auc']):.4f}")
        print(f"Mean Accuracy: {np.mean(results[arch]['acc']):.4f} ± {np.std(results[arch]['acc']):.4f}")

# Add this after your existing model training code (before the validation section)
print("\nTraining advanced neural architectures...")
X_scaled_neural = scaler.transform(X_resampled)
neural_results = cross_validate_neural_models(X_scaled_neural, y_resampled)
print_neural_results(neural_results)
