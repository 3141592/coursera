# 4.3 Predicting house prices: A regression example
# Listing 4.23 Loading the Boston housing dataset
from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(f"train_data.shape: {train_data.shape}")
print(f"test_data.shape: {test_data.shape}")
#print(f"train_targets: {train_targets}")

# 4.3.2 Preparing the data
# Listing 4.24 Normalizing data
mean = train_data.mean(axis=0)
print(f"train_data[0]: {train_data[0]}")
train_data -= mean
print(f"train_data[0]: {train_data[0]}")
std = train_data.std(axis=0)
train_data /= std
print(f"train_data[0]: {train_data[0]}")
test_data -= mean
test_data /= std

# 4.3.3 Building your model
# Listing 4.25 Model definition
def build_model():
    model = keras.Sequential([
        layers.dense(64, activtion="relu"),
        layers.dense(64, activtion="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop",
            loss="mse",
            metrics=["mae"])
    return model

# 4.34 Validating your approach using K-fold validation
# Listing 4.26 K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples],
             train_data[(i + 1) * num_val_samples:]],
            axis=0)

