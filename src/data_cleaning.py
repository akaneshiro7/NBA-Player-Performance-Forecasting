import pandas as pd
from sklearn.preprocessing import StandardScaler

starting_data = pd.read_csv("data/starting_data.csv")
starting_data.head()

for col in starting_data.columns:
    if col != 'player':
        starting_data[col] = pd.to_numeric(starting_data[col], errors='coerce')

# Remove Playesr with less than 20 games in a season
filtered_data = starting_data[starting_data['g'] >= 20]

# Remove Playesr with less than 20 games in a season
filtered_data = starting_data[starting_data['g'] >= 20]
filtered_data = filtered_data.groupby('player_id').filter(lambda x: len(x) >= 4)

# Sorting the data by player ID and then by experience
sorted_data = filtered_data.sort_values(by=['player_id', 'experience'])
sorted_data = sorted_data.drop("player", axis=1)

# Creating lagged statistics for 1 year and 2 years
lag1 = sorted_data.groupby('player_id').shift(1)
lag2 = sorted_data.groupby('player_id').shift(2)
lag3 = sorted_data.groupby('player_id').shift(3)

# Renaming columns for the lagged datasets
lag1.columns = [f"{col}_lag1" for col in lag1.columns]
lag2.columns = [f"{col}_lag2" for col in lag2.columns]
lag3.columns = [f"{col}_lag3" for col in lag3.columns]

# Concatenating the current, lag1, and lag2 data
lagged_data = pd.concat([sorted_data, lag1, lag2, lag3], axis=1)


# Dropping all rows with NaN values
cleaned_data = lagged_data.dropna()


unique_age_data = cleaned_data[
    (cleaned_data['age'] != cleaned_data['age_lag1']) & 
    (cleaned_data['age_lag1'] != cleaned_data['age_lag2']) &
    (cleaned_data['age_lag2'] != cleaned_data['age_lag3'])
]

numerical_features = unique_age_data.columns.drop(['player_id'])

scaler = StandardScaler()

unique_age_data[numerical_features] = scaler.fit_transform(unique_age_data[numerical_features])
unique_age_data.to_csv("data/cleaned_data.csv", index=False)
