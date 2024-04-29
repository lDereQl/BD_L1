import pandas as pd
import numpy as np

# Load training and testing data
train_df = pd.read_csv('train_20240206.csv')
test_df = pd.read_csv('test_20240206.csv')
test_df = test_df.loc[[0]]
# Fill missing values with median for continuous variables
continuous_variables = ['LotArea', 'SalePrice', '1stFlrSF', 'TotRmsAbvGrd', 'GrLivArea', 'LotFrontage']
for var in continuous_variables:
    train_df[var] = train_df[var].fillna(train_df[var].median())
    test_df[var] = test_df[var].fillna(train_df[var].median())

# Fill missing values with mode for categorical variables
categorical_variables = ['HouseStyle', 'Street']
for var in categorical_variables:
    train_df[var] = train_df[var].fillna(train_df[var].mode()[0])
    test_df[var] = test_df[var].fillna(train_df[var].mode()[0])

# Preprocess training data
selected_features = continuous_variables + categorical_variables
train_df_new = train_df[selected_features]
train_df_main = train_df_new.dropna()

# Preprocess testing data
test_df_new = test_df[selected_features]
test_df_main = test_df_new.dropna()

# One-hot encoding for categorical variables
train_df_main = pd.get_dummies(train_df_main, columns=categorical_variables)
test_df_main = pd.get_dummies(test_df_main, columns=categorical_variables)

# Align columns in training and testing datasets
train_columns = set(train_df_main.columns)
test_columns = set(test_df_main.columns)
missing_columns_in_test = train_columns - test_columns
missing_columns_in_train = test_columns - train_columns

for col in missing_columns_in_test:
    test_df_main[col] = 0

for col in missing_columns_in_train:
    train_df_main[col] = 0

# Standardize data
def standardize_data(processed_dataset, initial_dataset, column_name):
    mean = initial_dataset[column_name].mean()
    std = initial_dataset[column_name].std()
    processed_dataset[column_name] = (initial_dataset[column_name] - mean) / std
    return mean, std

# Standardize training data
prepared_train_data = train_df_main.copy()
means = {}
stds = {}
for feature in continuous_variables:
    mean, std = standardize_data(prepared_train_data, train_df_main, feature)
    means[feature] = mean
    stds[feature] = std

# Standardize testing data
prepared_test_data = test_df_main.copy()
for feature in continuous_variables:
    mean = means[feature]
    std = stds[feature]
    prepared_test_data[feature] = (test_df_main[feature] - mean) / std

# KNN implementation
def knn(train_data, test_instance, k):
    distances_indices = []
    for i in range(len(train_data)):
        train_instance = train_data.iloc[i, :-1].to_numpy(dtype=np.float64)  # Convert to NumPy array
        test_instance_np = test_instance.to_numpy(dtype=np.float64)  # Convert to NumPy array
        distance = np.linalg.norm(train_instance - test_instance_np)
        distances_indices.append((distance, i))
    sorted_distances_indices = sorted(distances_indices)[:k]
    indices = [index for distance, index in sorted_distances_indices]
    df_nearest = train_data.iloc[indices]
    return df_nearest

def predict_single(test_instance, train_data, k):
    nearest_neighbors = knn(train_data, test_instance, k)
    if nearest_neighbors.empty:
        return np.nan, np.nan  # Return NaN if no nearest neighbors are found
    prediction = nearest_neighbors["LotArea"].mean()  # Take mean of LotArea of nearest neighbors
    return prediction, test_instance["LotArea"]

# Define K value
K = 500

# Make predictions for test instances
predictions = []
actual_values = []
for i in range(1):  # Limit processing to the first 100 elements
    test_instance = prepared_test_data.iloc[i, :-1]  # Exclude the last column (target variable)
    prediction, actual = predict_single(test_instance, prepared_train_data, K)
    predictions.append(prediction * stds['LotArea'] + means['LotArea'])  # Reverse standardization
    actual_values.append(actual * stds['LotArea'] + means['LotArea'])  # Reverse standardization
    print(f"Test Instance {i + 1}: Predicted LotArea = {predictions[-1]:.2f}, Actual LotArea = {actual_values[-1]:.2f}")

# Calculate RMSE
def calculate_rmse(predictions, actual_values):
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    squared_errors = (predictions - actual_values) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    return rmse

# Calculate RMSE for the predictions
rmse = calculate_rmse(predictions, actual_values)
print("Root Mean Squared Error (RMSE):", rmse)
