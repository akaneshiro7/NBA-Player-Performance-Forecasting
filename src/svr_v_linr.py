import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt

total_df = pd.read_csv("data/cleaned_data.csv")

columns_to_drop = [
    "player_id", "age", "experience", "g", "mp_per_game", 
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

print(total_df)

# Features to be used in parameter estimation
features_df = total_df.drop(columns=columns_to_drop)

# Parameters to be estimated
targets_df = total_df[["player_id", "age", "experience", 
                       'pts_per_game', 'ast_per_game', 'orb_per_game', 
                       'drb_per_game']] 

# Normalizes inputs for PCA work
scaler = StandardScaler()


# Normalizes inputs for PCA work
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Output categories and PCA proportions
output_cats = ["pts_per_game", "orb_per_game", "drb_per_game", "ast_per_game"]
pca_proportions = [0.8, 0.9, 0.95, 0.99]

# Iterate through each output category
for output_cat in output_cats:
    plt.figure(figsize=(14, 24))  # Adjust the figure size as needed
    y = targets_df[output_cat]

    # Iterate through each PCA proportion
    for i, proportion in enumerate(pca_proportions):
        pca = PCA(n_components=proportion)
        pca_features = pca.fit_transform(scaled_features)

        x_train, x_test, y_train, y_test = train_test_split(pca_features, y, test_size=0.2, random_state=42)

        # SVR Model
        svr = SVR(kernel='rbf')
        svr.fit(x_train, y_train)
        y_pred_svr = svr.predict(x_test)
        rmse_svr = round(sqrt(mean_squared_error(y_test, y_pred_svr)), 3)
        r2_svr = round(r2_score(y_test, y_pred_svr), 3)

        # Linear Regression Model
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        y_pred_lr = lr.predict(x_test)
        rmse_lr = round(sqrt(mean_squared_error(y_test, y_pred_lr)), 3)
        r2_lr = round(r2_score(y_test, y_pred_lr), 3)

        # Plotting SVR
        plt.subplot(8, 2, 2*i + 1)  # 8 rows, 2 columns, current subplot position for SVR
        plt.scatter(y_test, y_pred_svr, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.title(f'SVR: {output_cat}, PCA: {proportion} \nRMSE: {rmse_svr}, R\u00B2: {r2_svr}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

        # Plotting Linear Regression
        plt.subplot(8, 2, 2*i + 2)  # Position for Linear Regression
        plt.scatter(y_test, y_pred_lr, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.title(f'Linear Regression: {output_cat}, PCA: {proportion} \nRMSE: {rmse_lr}, R\u00B2: {r2_lr}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')

    plt.tight_layout()
    plt.show()
