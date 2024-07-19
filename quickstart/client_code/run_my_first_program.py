import numpy as np
from sklearn.linear_model import LinearRegression
from nada_dsl import *

# Data augmentation using random noise
def generate_synthetic_data(num_samples=1000, num_features=10):
    base_data = np.random.randn(num_samples, num_features)
    noise = np.random.normal(0, 0.1, base_data.shape)
    synthetic_data = base_data + noise
    return synthetic_data

# Train the Linear Regression model
def train_linear_regression(X, y):
    model = LinearRegression()
    fit_model = model.fit(X, y)
    return fit_model

# Generate synthetic input data for inference
def generate_synthetic_input(num_features=10):
    return np.random.normal(0, 1, (num_features,))

# Main function to integrate everything
def main():
    # Generate synthetic data
    num_features = 10
    X_augmented = generate_synthetic_data(1000, num_features)

    # Define the expected weights and bias for the linear regression
    expected_weights = np.ones((num_features,))
    expected_bias = 4.2

    # Generate synthetic labels
    y_augmented = X_augmented @ expected_weights + expected_bias

    # Train the linear regression model
    fit_model = train_linear_regression(X_augmented, y_augmented)

    # Generate synthetic input data for inference
    synthetic_input = generate_synthetic_input(num_features)

    # Perform inference using the trained model
    prediction = fit_model.predict([synthetic_input])

    print(f"Generated Input: {synthetic_input}")
    print(f"Model Output: {prediction}")

    # Store the vector using Nada DSL
    vector = prediction[0]
    storage_party = Party(name="Predictions_Data")
    secret_vector = SecretInteger(Input(name="results", party=storage_party))
    stored_vector = Output(secret_vector, "results", storage_party)

    return stored_vector

# Execute the main function
stored_results = main()
print(stored_results)
