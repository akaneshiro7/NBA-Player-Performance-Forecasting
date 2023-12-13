import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def generate_log_reg_plots(choice):

    total_df = pd.read_csv("data/cleaned_data.csv")

    # Computes year-to-year changes in output categories
    delta_points = (total_df["pts_per_game"] - total_df["pts_per_game_lag1"]).tolist()
    delta_orebs = (total_df["orb_per_game"] - total_df["orb_per_game_lag1"]).tolist()
    delta_drebs = (total_df["drb_per_game"] - total_df["drb_per_game_lag1"]).tolist()
    delta_asts = (total_df["ast_per_game"] - total_df["ast_per_game_lag1"]).tolist()

    # Constructs parameters dataframe
    params_df = pd.DataFrame()
    params_df["player_id"] = total_df["player_id"]
    params_df["age"] = total_df["age"]
    params_df["experience"] = total_df["experience"]
    params_df["pts_delta"] = delta_points
    params_df["orb_delta"] = delta_orebs
    params_df["drb_delta"] = delta_drebs
    params_df["ast_delta"] = delta_asts

    # Adds output params
    total_df["pts_delta"] = delta_points
    total_df["orb_delta"] = delta_orebs
    total_df["drb_delta"] = delta_drebs
    total_df["ast_delta"] = delta_asts



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
        "percent_corner_3s_of_3pa", "corner_3_point_percent",
        "pts_delta", "orb_delta", "drb_delta", "ast_delta"
    ]

    # Features to be used in parameter estimation



    # 1 = better output than last year
    # 0 = worse output than last year

    cats_list = ['pts_delta', 'orb_delta', 'drb_delta', 'ast_delta']
    choice1_cats_list = ["pts_per_game", "orb_per_game", "drb_per_game", "ast_per_game"]
    choice2_cats_list = ["pts_per_game_lag1", "orb_per_game_lag1", "drb_per_game_lag1", "ast_per_game_lag1"]

    

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = axs.flatten()  # Flatten the array to easily iterate over it
    fig.suptitle("")

    fig2, axs2 = plt.subplots(2, 2, figsize=(12, 12))
    axs2 = axs2.flatten()  
    fig2.suptitle("")


    # Sets figure titles
    if choice == 0:
        title_1 = "ROC Curves of Points, Rebounds, and Assists of All Players"
        title_2 = "Points, Rebounds, and Assists v. Age of All Players"
    elif choice == 1:
        title_1 = "ROC Curves of Points, Rebounds, and Assists of Regressing/Breakout Players"
        title_2 = "Points, Rebounds, and Assists v. Age of Regressing/Breakout Players"
    elif choice == 2:
        title_1 = "ROC Curves of Points, Rebounds, and Assists of Young and Old Players"
        title_2 = "Points, Rebounds, and Assists v. Age of Young and Old Players"
    
    fig.suptitle(title_1)
    fig2.suptitle(title_2)

    for i in range(len(cats_list)):


        # Case where players over/underperformed by 25% of previous seasons' totals
        if choice == 1:
            # Filter total_df based on the condition
            condition = abs(total_df[cats_list[i]]) > (0.25 * total_df[choice2_cats_list[i]])
            filtered_df = total_df[condition]

            # Filter params_df using the same condition
            filtered_params = params_df[condition]

            # Prepare features and target variable for logistic regression
            features_df = filtered_df.drop(columns=columns_to_drop)
            delta_binary = (filtered_params[cats_list[i]] > 0).astype(int)

            plot_df = filtered_df[["age", choice1_cats_list[i]]]
            plot_df["delta_binary"] = delta_binary

        elif choice == 2:
            # Filter for players aged under 25 or over 32
            filtered_df = total_df[(total_df["age"] <= 25) | (total_df["age"] >= 32)]

            # Prepare features and target variable for logistic regression
            features_df = filtered_df.drop(columns=columns_to_drop)

            delta_binary = (params_df.loc[filtered_df.index, cats_list[i]] > 0).astype(int)

            plot_df = total_df[["age", choice1_cats_list[i]]]
            plot_df["delta_binary"] = delta_binary


        else:
            # Default case: use all data
            features_df = total_df.drop(columns=columns_to_drop)
            delta_binary = (params_df[cats_list[i]] > 0).astype(int)

            plot_df = total_df[["age", choice1_cats_list[i]]]
            plot_df["delta_binary"] = delta_binary


            

        # Calculates class priors based on total dataset
        improved_class_prior = round((delta_binary.sum() / len(delta_binary)), 3)
        declined_class_prior = 1 - improved_class_prior

        

        # Normalizes features during test
        scaler = StandardScaler()
        features_df = scaler.fit_transform(features_df)


        # Splits training and testing data
        x_train, x_test, y_train, y_test = train_test_split(features_df, 
                                                            delta_binary, 
                                                            test_size=0.2, 
                                                            random_state=42)

        # Creates logistic regression model with 1000 iterations
        log_reg_model = LogisticRegression(max_iter=1000)
        log_reg_model.fit(x_train, y_train)

        # Checks model performance on test set
        y_pred = log_reg_model.predict_proba(x_test)[:, 1]

        # Generates true positive and true negative 
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        # Empirically calculates the minimum probability of error along the ROC curve
        distances = np.sqrt(fpr**2 + (1 - tpr)**2)
        min_distance_index = np.argmin(distances)
        optimal_fpr, optimal_tpr = fpr[min_distance_index], tpr[min_distance_index]
        min_prob_error = round(((optimal_fpr * declined_class_prior) + ((1 - optimal_tpr) * improved_class_prior)), 3)

        # Plots output ROC
        axs[i].plot(fpr, tpr, color='blue', label='ROC Curve', linewidth=3)
        axs[i].plot(optimal_fpr, optimal_tpr, marker='o', color='red', label=f'Optimal Pe={min_prob_error}', markersize=15)
        axs[i].plot([0, 1], [0, 1], color='darkgrey', linestyle='--')
        axs[i].set_xlabel('False Positive Rate')
        axs[i].set_ylabel('True Positive Rate')
        axs[i].set_title(f'ROC Curve of {cats_list[i]}')
        axs[i].legend(loc="lower right")
        axs[i].grid(True)

        # Plots general categories v. age
        over_subset = plot_df[plot_df["delta_binary"] == 1]
        under_subset = plot_df[plot_df["delta_binary"] == 0]
        axs2[i].scatter(under_subset['age'], under_subset[choice1_cats_list[i]], c="red", s=10, alpha=0.25, label="Underperformed")
        axs2[i].scatter(over_subset['age'], over_subset[choice1_cats_list[i]], c="green", s=10, alpha=0.25, label="Overperformed")

        axs2[i].set_title(f"{cats_list[i]}, N={len(plot_df[choice1_cats_list[i]])}")
        axs2[i].set_xlabel('Age')
        axs2[i].set_ylabel(choice1_cats_list[i])
        axs2[i].legend()


    # Outputs plots
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":

    generate_log_reg_plots(0)
    generate_log_reg_plots(1)
    generate_log_reg_plots(2)