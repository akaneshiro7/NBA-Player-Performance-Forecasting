import numpy as np
import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/cleaned_data.csv")
# Separating features and targets
columns_to_drop = [
    "player_id", "g", "mp_per_game", 
    "fg_per_game", "fga_per_game", "ft_per_game", "fta_per_game",
    "orb_per_game", "drb_per_game", "ast_per_game", "tov_per_game",
    "pf_per_game", "pts_per_game", "orb_percent", "drb_percent", 
    "ast_percent", "tov_percent", "usg_percent", "avg_dist_fga", 
    "percent_fga_from_x0_3_range", "percent_fga_from_x3_10_range", 
    "percent_fga_from_x10_16_range", "percent_fga_from_x16_3p_range", 
    "percent_fga_from_x3p_range", "fg_percent_from_x0_3_range", 
    "fg_percent_from_x3_10_range", "fg_percent_from_x10_16_range", 
    "fg_percent_from_x16_3p_range", "fg_percent_from_x3p_range", 
    "percent_assisted_x2p_fg", "percent_assisted_x3p_fg", 
    "percent_corner_3s_of_3pa", "corner_3_point_percent"
]

features = data.drop(columns=columns_to_drop)
targets = data[['pts_per_game', 'ast_percent', 'orb_per_game', 'drb_per_game']]

# Standardizing the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Applying PCA
pca = PCA(n_components=0.95)  # Keeping 95% of the variance
features_pca = pca.fit_transform(features)

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

# Plotting PCA Variance Explained
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.title('PCA Variance Explained')
plt.show()

# Plotting Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()


# Predictions
predictions = model.predict(X_test)

# Plotting Predicted vs Actual for each target
for i, target in enumerate(targets.columns):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test[target], predictions[:, i], alpha=0.3)
    plt.xlabel('Actual Values')
    plt.ylabel('Predictions')
    plt.title(f'Predicted vs Actual for {target}')
    plt.show()
