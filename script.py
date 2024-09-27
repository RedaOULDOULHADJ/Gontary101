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

    # Drop the 'Id' column
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
    X= scaler.fit_transform(X)
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return "Data preparation successful.", (X_train, X_test, y_train, y_test, encoder, ordinal_encoder, scaler, poly)

# Example usage:
filepath = 'train.csv'
message, prepared_data = prepare_house_price_data(filepath)

if prepared_data:
    X_train, X_test, y_train, y_test, encoder, ordinal_encoder, scaler, poly = prepared_data
    print(message)
else:
    print(message)

# 2. Implementing Regularized Gradient Descent (with L2 Regularization)
def hypothesis(X, theta):
    return np.dot(X, theta)

def compute_cost_with_l2(X, y, theta, lambda_):
    m = len(y)
    predictions = hypothesis(X, theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    
    # Add the regularization term (exclude the bias term)
    regularization = (lambda_ / (2 * m)) * np.sum(np.clip(theta[1:] ** 2, -1e5, 1e5))  # Clipping large values

    
    return cost + regularization

def gradient_descent_with_l2(X, y, theta, alpha, iterations, lambda_):
    m = len(y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        predictions = hypothesis(X, theta)
        errors = predictions - y
        gradient = (1 / m) * np.dot(X.T, errors)
        gradient = np.clip(gradient, -1e5, 1e5)
        # Apply L2 regularization (skip the first term for bias)
        theta_reg = theta.copy()
        theta_reg[0] = 0  # Do not regularize the bias term
        regularization_term = (lambda_ / m) * theta_reg

        # Update theta (parameters)
        theta = theta - alpha * (gradient + regularization_term)
        
        # Save the cost for this iteration (including L2 regularization in cost function)
        cost_history[i] = compute_cost_with_l2(X, y, theta, lambda_)
    
    return theta, cost_history

# Example usage of linear regression with L2 regularization
alpha = 0.002  # Learning rate
iterations = 1000  # Number of iterations
theta = np.zeros(X_train.shape[1])  # One weight per feature
lambda_ = 0.001  # Regularization parameter

# Run gradient descent
#theta_final, cost_history = gradient_descent_with_l2(X_train, y_train, theta, alpha, iterations, lambda_)

# 3. Predict and Evaluate the Model
def predict(X, theta):
    return hypothesis(X, theta)  # hypothesis = X.dot(theta)

# Predictions on the test dataset
#y_pred = predict(X_test, theta_final)

# Evaluate the model using various metrics
#mse = mean_squared_error(y_test, y_pred)
#rmse = np.sqrt(mse)
##r2 = r2_score(y_test, y_pred)

##print(f"Test MSE: {mse}")
##print(f"Test RMSE: {rmse}")
##print(f"R² Score: {r2}")

# 4. Cross-Validation
def cross_validation(X_train, y_train):
    from sklearn.linear_model import Ridge
    ridge_model = Ridge(alpha=lambda_)  # Using Ridge regression for L2 regularization

    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(ridge_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    rmse_cv = np.sqrt(-cv_scores)
    print(f"Cross-Validation RMSE: {rmse_cv.mean()} ± {rmse_cv.std()}")

#cross_validation(X_train, y_train)

# 5. Visualization of Predicted vs Actual Values


# 6. Testing a More Complex Model (Random Forest)
rf_model = RandomForestRegressor(n_estimators=70, random_state=37, max_features=int(X_train.shape[1]), max_samples = 500 )
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_train)

plt.scatter(y_train, y_pred_rf, alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Predicted vs. Actual House Prices")
# Add the y=x line
min_val = min(min(y_train), min(y_pred_rf))
max_val = max(max(y_train), max(y_pred_rf))
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x') 

plt.show()
# Evaluate Random Forest model
mse_rf = mean_squared_error(y_train, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
r2_rf = r2_score(y_train, y_pred_rf)

print(f"Random Forest Test MSE: {mse_rf}")
print(f"Random Forest Test RMSE: {rmse_rf}")
print(f"Random Forest R² Score: {r2_rf}")


