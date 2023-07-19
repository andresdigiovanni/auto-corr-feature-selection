import pandas as pd

from auto_corr_feature_selection import AutoCorrFeatureSelection

# download data
dataset_url = "https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv"
iris_df = pd.read_csv(dataset_url)

# set up auto correlation
auto_corr = AutoCorrFeatureSelection(iris_df)

# select low correlated columns
selected_columns = auto_corr.select_columns_above_threshold(threshold=0.85)
iris_filtered_df = iris_df[selected_columns]

# summary
print("Number of original columns: {}".format(len(iris_df.columns)))
print("Number of selected columns: {}\n".format(len(iris_filtered_df.columns)))

print("Selected columns: {}\n".format(selected_columns))

correlation_matrix = auto_corr.correlation_matrix()
print("Correlation matrix:")
print(correlation_matrix)
