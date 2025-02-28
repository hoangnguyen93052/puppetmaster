import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        if os.path.exists(self.file_path):
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}.")
        else:
            print(f"File {self.file_path} does not exist.")
            raise FileNotFoundError(f"Cannot find {self.file_path}.")

    def clean_data(self):
        initial_shape = self.data.shape
        self.data.dropna(inplace=True)
        self.data = self.data.reset_index(drop=True)
        final_shape = self.data.shape
        print(f"Cleaned data: {initial_shape} -> {final_shape}")
    
    def save_cleaned_data(self, output_path):
        self.data.to_csv(output_path, index=False)
        print(f"Cleaned data saved to {output_path}.")

    def transform_data(self):
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        
        # Encoding categorical variables
        for col in categorical_cols:
            if self.data[col].nunique() < 10:
                self.data[col] = self.data[col].astype('category').cat.codes
        
        print(f"Transformed categorical columns: {categorical_cols.tolist()}")

        # Normalizing numerical variables
        scaler = StandardScaler()
        self.data[numerical_cols] = scaler.fit_transform(self.data[numerical_cols])
        print(f"Normalized numerical columns: {numerical_cols.tolist()}")

    def analyze_data(self):
        print("Analyzing data...")
        correlation_matrix = self.data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation Heatmap')
        plt.savefig('correlation_heatmap.png')
        plt.show()

    def split_data(self, target_column):
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Data split into training and testing sets. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return X_train, X_test, y_train, y_test

def main():
    file_path = 'data.csv'
    output_path = 'cleaned_data.csv'
    
    processor = DataProcessor(file_path)
    processor.load_data()
    processor.clean_data()
    processor.save_cleaned_data(output_path)
    processor.transform_data()
    processor.analyze_data()
    
    target_column = 'target'  # Change this to your target column
    X_train, X_test, y_train, y_test = processor.split_data(target_column)

if __name__ == "__main__":
    main()