import numpy as np
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

data = pd.read_csv("data/cleaned_data.csv")
# Separating features and targets
features = data.drop(['pts_per_game', 'ast_percent', 'orb_per_game', 'drb_per_game'], axis=1)
targets = data[['pts_per_game', 'ast_percent', 'orb_per_game', 'drb_per_game']]

# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Applying PCA
pca = PCA(n_components=0.95)  # Keeping 95% of the variance
features_pca = pca.fit_transform(features_scaled)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_pca, targets, test_size=0.2, random_state=42)

# Displaying the results
print(features_pca.shape, pca.explained_variance_ratio_.sum())  # New shape of features and total explained variance

# Removing non-numeric and irrelevant columns
features = data.select_dtypes(include=[np.number]).drop(['player_id', 'pts_per_game', 'ast_percent', 'orb_per_game', 'drb_per_game'], axis=1)

# Standardizing the features again
features_scaled = scaler.fit_transform(features)

# Applying PCA again
features_pca = pca.fit_transform(features_scaled)

# Splitting the data into training and testing sets again
X_train, X_test, y_train, y_test = train_test_split(features_pca, targets, test_size=0.2, random_state=42)

# New shape of features and total explained variance
features_pca.shape, pca.explained_variance_ratio_.sum() 

# Neural network architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(4)  # Output layer for 4 targets
])

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32)

# Evaluating the model on the test set
test_loss = model.evaluate(X_test, y_test)

print(test_loss)
# numerical_features = data.columns.drop(['player_id'])

# # Applying PCA
# # Excluding non-numerical columns for PCA
# pca_features = data[numerical_features]

# # Initialize PCA, let's start by keeping 95% of the variance
# pca = PCA(n_components=0.95)
# pca_data = pca.fit_transform(pca_features)

# # Checking the number of components selected by PCA
# n_components = pca.n_components_
# variance_explained = np.sum(pca.explained_variance_ratio_)

# print(n_components, variance_explained)

# pdb.set_trace()

# # Data Preparation for Neural Network
# # Since we need to predict points, rebounds, and assists, let's assume these are the last three columns in our PCA data
# # Splitting the PCA data into features (X) and targets (y)
# X = pca_data[:, :-3]
# y = pca_data[:, -3:]

# # Splitting the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Model Architecture
# model = Sequential()
# model.add(Dense(64, input_dim=51, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(3))  # 3 outputs for points, rebounds, and assists

# # Compilation
# model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# # Training
# # Due to computational limitations, we'll use a small number of epochs
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32)

# # The detailed training process and performance evaluation will require more resources and fine-tuning
# # print(history.history)
