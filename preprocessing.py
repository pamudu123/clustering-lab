import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def clean_unrealistic_values(data):
    """
    Remove rows with unrealistic values based on medical domain knowledge.
    """
    unrealistic_criteria = {
        'age': lambda x: (x <= 0) | (x > 120),
        'blood_pressure': lambda x: (x <= 30) | (x > 270),
        'cholesterol': lambda x: (x <= 0) | (x > 500),
        'max_heart_rate': lambda x: (x <= 25) | (x > 220),
        'plasma_glucose': lambda x: x <= 0,
        'skin_thickness': lambda x: x < 0,
        'insulin': lambda x: x < 0,
        'bmi': lambda x: (x <= 0) | (x > 70),
        'diabetes_pedigree': lambda x: x < 0
    }
    mask = pd.Series(True, index=data.index)
    for column, condition in unrealistic_criteria.items():
        if column in data.columns:
            unrealistic_values = condition(data[column])
            mask = mask & ~unrealistic_values
    cleaned_data = data[mask].copy()
    return cleaned_data

def preprocess_data(df):
    """
    Preprocess the training data for clustering with one-hot encoding
    for categorical features and returns the scaler and encoded column names.
    """
    df_processed = df.copy()
    patient_ids = None
    if 'patient_id' in df_processed.columns:
        patient_ids = df_processed['patient_id'].copy()
        df_processed.drop('patient_id', axis=1, inplace=True)
    all_numerical_cols = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate',
                          'plasma_glucose', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree']
    all_categorical_cols = ['gender', 'chest_pain_type', 'exercise_angina',
                           'hypertension', 'heart_disease', 'residence_type', 'smoking_status']
    numerical_cols = [col for col in all_numerical_cols if col in df_processed.columns]
    categorical_cols = [col for col in all_categorical_cols if col in df_processed.columns]
    scaler = StandardScaler()
    if numerical_cols:
        df_processed[numerical_cols] = scaler.fit_transform(df_processed[numerical_cols])
    original_cols = df_processed.columns.tolist()
    if categorical_cols:
        df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=False, dtype=int)
    encoded_cols = df_processed.columns.tolist()
    return df_processed, numerical_cols, encoded_cols, scaler, patient_ids

def preprocess_test_data(df_test, scaler, train_numerical_cols, train_encoded_cols):
    """
    Preprocess the test data using the scaler fitted on the training data
    and ensures consistent one-hot encoding.
    """
    df_test_processed = df_test.copy()
    patient_ids = None
    if 'patient_id' in df_test_processed.columns:
        patient_ids = df_test_processed['patient_id'].copy()
        df_test_processed.drop('patient_id', axis=1, inplace=True)
    all_categorical_cols = ['gender', 'chest_pain_type', 'exercise_angina',
                           'hypertension', 'heart_disease', 'residence_type', 'smoking_status']
    test_numerical_cols = [col for col in train_numerical_cols if col in df_test_processed.columns]
    test_categorical_cols = [col for col in all_categorical_cols if col in df_test_processed.columns]
    if test_numerical_cols:
        cols_to_scale = [col for col in test_numerical_cols if col in df_test_processed.columns]
        if cols_to_scale:
            df_test_processed[cols_to_scale] = scaler.transform(df_test_processed[cols_to_scale])
    if test_categorical_cols:
        df_test_processed = pd.get_dummies(df_test_processed, columns=test_categorical_cols, drop_first=False, dtype=int)
    current_test_cols = df_test_processed.columns.tolist()
    for col in train_encoded_cols:
        if col not in current_test_cols:
            df_test_processed[col] = 0
    df_test_processed = df_test_processed[train_encoded_cols]
    return df_test_processed, patient_ids
