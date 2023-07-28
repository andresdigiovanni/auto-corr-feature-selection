import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from auto_corr_feature_selection import AutoCorrFeatureSelection

# download data
iris = load_iris()
iris_df = pd.DataFrame(
    data=np.c_[iris["data"], iris["target"]], columns=iris["feature_names"] + ["target"]
)

# set up auto correlation
auto_corr = AutoCorrFeatureSelection(iris_df.drop(columns=["target"]))

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
