import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv("data/cleaned_data.csv")

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)  # Using 2 components for illustration, can be adjusted
pca_result = pca.fit_transform(scaled_data)

# Convert the PCA result into a DataFrame for better visualization
pca_df = pd.DataFrame(data=pca_result, columns=['Principal Component 1', 'Principal Component 2'])

# Save the PCA result to a new CSV file
output_file_path = 'data/PCAOutput.csv'  # Replace with your desired output file path
pca_df.to_csv(output_file_path, index=False)

print(pca_df.head())
