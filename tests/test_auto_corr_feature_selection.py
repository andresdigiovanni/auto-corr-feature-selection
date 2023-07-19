import numpy as np
import pandas as pd

from auto_corr_feature_selection.auto_corr_feature_selection import (
    AutoCorrFeatureSelection,
)


# Tests that the correlation matrix is calculated correctly
def test_correlation_matrix_calculation():

    # Arrange
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Act
    auto_corr = AutoCorrFeatureSelection(data)

    # Assert
    expected = pd.DataFrame(
        {"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0], "C": [1.0, 1.0, 1.0]},
        index=["A", "B", "C"],
    )
    result = auto_corr.correlation_matrix()

    assert result.equals(expected)


# Tests that select_columns_above_threshold returns the correct columns
def test_select_columns_above_threshold():

    # Arrange
    data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})

    # Act
    auto_corr = AutoCorrFeatureSelection(data)

    # Assert
    expected = ["C"]
    result = auto_corr.select_columns_above_threshold()

    assert result == expected


# Tests that the function handles an empty dataframe
def test_empty_dataframe_handling():

    # Arrange
    data = pd.DataFrame()

    # Act
    auto_corr = AutoCorrFeatureSelection(data)

    # Assert
    expected = []
    result = auto_corr.select_columns_above_threshold()

    assert result == expected


# Tests that the function handles a dataframe with only one column
def test_single_column_dataframe_handling():

    # Arrange
    data = pd.DataFrame({"A": [1, 2, 3]})

    # Act
    auto_corr = AutoCorrFeatureSelection(data)

    # Assert
    expected = ["A"]
    result = auto_corr.select_columns_above_threshold()

    assert result == expected


# Tests that the function handles a dataframe with only string columns
def test_string_dataframe_handling():

    # Arrange
    data = pd.DataFrame(
        {"A": ["a", "b", "c"], "B": ["d", "e", "f"], "C": ["g", "h", "i"]}
    )

    # Act
    auto_corr = AutoCorrFeatureSelection(data)

    # Assert
    expected = ["C"]
    result = auto_corr.select_columns_above_threshold()

    assert result == expected


# Tests that the function handles a dataframe with missing values
def test_missing_values_handling():

    # Arrange
    data = pd.DataFrame({"A": [1, 2, np.nan], "B": [4, np.nan, 6], "C": [7, 8, 9]})

    # Act
    auto_corr = AutoCorrFeatureSelection(data)

    # Assert
    expected = ["C"]
    result = auto_corr.select_columns_above_threshold()

    assert result == expected
