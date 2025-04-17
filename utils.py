import pandas as pd
from typing import List, Tuple, Optional

def get_column_lists(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Return lists of numerical and categorical column names present in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple[List[str], List[str]]: Numerical and categorical column lists.
    """
    all_numerical_cols = ['age', 'blood_pressure', 'cholesterol', 'max_heart_rate',
                          'plasma_glucose', 'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree']
    all_categorical_cols = ['gender', 'chest_pain_type', 'exercise_angina',
                           'hypertension', 'heart_disease', 'residence_type', 'smoking_status']
    numerical_cols = [col for col in all_numerical_cols if col in df.columns]
    categorical_cols = [col for col in all_categorical_cols if col in df.columns]
    return numerical_cols, categorical_cols
