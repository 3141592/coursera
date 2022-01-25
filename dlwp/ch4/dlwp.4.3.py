# 4.3 Predicting house prices: A regression example

# Listing 4.23 Loading the Boston housing dataset
from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(f"train_data.shape: {train_data.shape}")
print(f"test_data.shape: {test_data.shape}")
print(f"train_targets: {train_targets}")

