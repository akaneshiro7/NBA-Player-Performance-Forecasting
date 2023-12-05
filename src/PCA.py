from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

df = pd.read_csv("data/cleaned_data.csv")

numerical_features = df.columns.drop(['player_id'])

# Applying PCA
# Excluding non-numerical columns for PCA
pca_features = df[numerical_features]

# Initialize PCA, let's start by keeping 95% of the variance
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(pca_features)

# Checking the number of components selected by PCA
n_components = pca.n_components_
variance_explained = np.sum(pca.explained_variance_ratio_)

print(n_components, variance_explained)



# Data Preparation for Neural Network
# Since we need to predict points, rebounds, and assists, let's assume these are the last three columns in our PCA data
# Splitting the PCA data into features (X) and targets (y)
X = pca_data[:, :-3]
y = pca_data[:, -3:]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Architecture
model = Sequential()
model.add(Dense(64, input_dim=51, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3))  # 3 outputs for points, rebounds, and assists

# Compilation
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Training
# Due to computational limitations, we'll use a small number of epochs
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1000, batch_size=32)

# The detailed training process and performance evaluation will require more resources and fine-tuning
print(history.history)
