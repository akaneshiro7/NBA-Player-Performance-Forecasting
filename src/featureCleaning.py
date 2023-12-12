import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/cleaned_data.csv")
# Separating features and targets
columns_to_drop = [
    "player_id", "g", "mp_per_game", 
    "fg_per_game", "fga_per_game", "ft_per_game", "fta_per_game",
    "tov_per_game",
    "pf_per_game",  "orb_percent", "drb_percent", 
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

features.to_csv("data/feature.csv")
