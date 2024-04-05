import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Function to load the dataset
@st.cache
def load_data():
    # Specify the correct path to the dataset file
    file_path = r"C:\Users\sonaw\Desktop\react1\dermatology_database_1.csv"

    df = pd.read_csv(file_path)
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    return df

# Main function
def main():
    st.title('Dermatology Diagnosis with SVM Classifier')
    
    # Load the dataset
    df = load_data()

    # Selecting features and target variable
    X = df.drop('class', axis=1)
    y = df['class']

    # Splitting the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Input section
    st.sidebar.header('Input Parameters')
    
    # Input for each feature
    input_features = []
    for feature in X.columns:
        input_features.append(st.sidebar.slider(f"Enter value for {feature}", float(X[feature].min()), float(X[feature].max())))
    
    # Convert input features to DataFrame
    input_data = pd.DataFrame([input_features], columns=X.columns)

    # Imputation
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    input_data_imputed = pd.DataFrame(imputer.transform(input_data), columns=input_data.columns)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(X_train)
    input_data_scaled = pd.DataFrame(scaler.transform(input_data_imputed), columns=input_data.columns)

    # Create and train SVM model
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train, y_train)

    # Prediction
    if st.button("Predict"):
        prediction = svm_classifier.predict(input_data_scaled)
        st.write('Prediction:')
        st.write(f'The predicted class is: {prediction}')

# Run the main function
if __name__ == "__main__":
    main()
