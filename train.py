import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier  # Updated import
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ignore warnings
warnings.filterwarnings('ignore')

# Set display options and plot style
plt.style.use('fivethirtyeight')
pd.set_option('display.max_columns', 26)

# Load and preprocess data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.drop('id', axis=1, inplace=True)
    df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                  'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                  'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                  'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                  'aanemia', 'class']
    return df

def clean_data(df):
    # Convert columns to numeric
    df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
    df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
    df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')
    
    # Fix inconsistent values in categorical columns
    df['diabetes_mellitus'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
    df['coronary_artery_disease'] = df['coronary_artery_disease'].replace('\tno', 'no')
    df['class'] = df['class'].replace({'ckd\t': 'ckd', 'notckd': 'not ckd'})
    
    # Map target column
    df['class'] = df['class'].map({'ckd': 0, 'not ckd': 1})
    df['class'] = pd.to_numeric(df['class'], errors='coerce')
    
    return df

# Impute missing values
def random_value_imputation(df, feature):
    random_sample = df[feature].dropna().sample(df[feature].isna().sum())
    random_sample.index = df[df[feature].isnull()].index
    df.loc[df[feature].isnull(), feature] = random_sample

def impute_mode(df, feature):
    mode = df[feature].mode()[0]
    df[feature] = df[feature].fillna(mode)

def handle_missing_values(df):
    num_cols = [col for col in df.columns if df[col].dtype != 'object']
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    
    for col in num_cols:
        random_value_imputation(df, col)
        
    random_value_imputation(df, 'red_blood_cells')
    random_value_imputation(df, 'pus_cell')

    for col in cat_cols:
        impute_mode(df, col)
        
    return df

# Encode categorical variables
def encode_categorical(df):
    cat_cols = [col for col in df.columns if df[col].dtype == 'object']
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
    return df

# Train model
def train_model(X_train, y_train):
    rf = RandomForestClassifier()  # Updated model to RandomForestClassifier
    rf.fit(X_train, y_train)
    return rf

# Evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    class_report = classification_report(y_test, model.predict(X_test))

    print(f"Training Accuracy: {train_acc}")
    print(f"Test Accuracy: {test_acc}\n")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
    print(f"Classification Report:\n{class_report}")

# Main function
def main():
    # Load data
    filepath = 'kidney_disease.csv'
    df = load_data(filepath)

    # Clean and preprocess data
    df = clean_data(df)
    df = handle_missing_values(df)
    df = encode_categorical(df)

    # Prepare features and target
    ind_col = [col for col in df.columns if col != 'class']
    dep_col = 'class'
    X = df[ind_col]
    y = df[dep_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_train, X_test, y_train, y_test)

    # Save the model to a file
    with open('rf_model.pkl', 'wb') as file:  # Updated filename to rf_model.pkl
        pickle.dump(model, file)
    print("Model saved as rf_model.pkl")

# Run the main function
if __name__ == "__main__":
    main()




