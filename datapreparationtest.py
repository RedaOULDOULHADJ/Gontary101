import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 1. Data Preparation
def prepare_house_price_data(csv_filepath):
    try:
        # Load the dataset
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        return "Error: File not found.", None

    # Check the column names
    print("Available columns:", df.columns)

    # Ensure 'SalePrice' is in the DataFrame
    if "SalePrice" not in df.columns:
        return "Error: 'SalePrice' column not found.", None

    # Drop the 'Id' column if it exists
    if "Id" in df.columns:
        df.drop("Id", axis=1, inplace=True)

    # Separate features (X) and target (y)
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    # Identify numerical and categorical features
    numerical_cols = X.select_dtypes(include=np.number).columns
    categorical_cols = X.select_dtypes(exclude=np.number).columns

    # Impute missing numerical values with the median
    for col in numerical_cols:
        X[col] = X[col].fillna(X[col].median())

    # Impute missing categorical values with a placeholder ('Missing')
    for col in categorical_cols:
        X[col] = X[col].fillna("Missing")

    # Distinguish between columns with many categories vs few
    many_categories_cols = [col for col in categorical_cols if X[col].nunique() > 10]
    few_categories_cols = [col for col in categorical_cols if X[col].nunique() <= 10]

    # One-hot encode features with few categories
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_few_categorical = encoder.fit_transform(X[few_categories_cols])
    encoded_few_df = pd.DataFrame(encoded_few_categorical, columns=encoder.get_feature_names_out(few_categories_cols))
    
    # Ordinal encode features with many categories
    ordinal_encoder = OrdinalEncoder()
    encoded_many_categorical = ordinal_encoder.fit_transform(X[many_categories_cols])
    encoded_many_df = pd.DataFrame(encoded_many_categorical, columns=many_categories_cols)
    
    # Combine all features back
    X = pd.concat([X.drop(categorical_cols, axis=1), encoded_few_df, encoded_many_df], axis=1)

    # Scale numerical features only
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Apply polynomial features (degree 2) for non-linearity
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = poly.fit_transform(X)
    X = scaler.fit_transform(X)
    return X, y



X, Y= prepare_house_price_data("test.csv")
pd.DataFrame(X).to_csv("X_processed.csv", index=False)
pd.DataFrame(Y).to_csv("y_processed.csv", index=False)