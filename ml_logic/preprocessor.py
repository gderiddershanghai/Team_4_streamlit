import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_and_split(fp='../Data/watson_healthcare_modified.csv',
                         columns_to_use=None,
                         test_size=0.2,
                         random_state=1) -> tuple:
    """
    Preprocesses the given dataset by encoding binary categorical columns as 0 or 1,
    one-hot encoding other categorical columns, and splits it into training and testing sets
    with numeric features scaled after the split to prevent data leakage.
    'Attrition' is considered the target variable and is always included.

    Parameters:
    - fp (str, optional): File path to the dataset. Defaults to '../Data/watson_healthcare_modified.csv'.
    - columns_to_use (list of str, optional): Columns to be used. If None, all except 'EmployeeID' are used.
      'Attrition' is automatically included regardless of this list.
    - test_size (float, optional): Proportion of the dataset to include in the test split.
    - random_state (int, optional): Controls the shuffling applied to the data before applying the split.

    Returns:
    - tuple: A tuple containing split datasets (X_train, X_test, y_train, y_test).
    """

    # Load the dataset
    df = pd.read_csv(fp)
    df = df.drop_duplicates()

    # Ensure 'Attrition' is always included
    mandatory_columns = ['Attrition']

    # If columns_to_use is None, use all columns except 'EmployeeID', ensuring 'Attrition' is included
    if columns_to_use is None:
        columns_to_use = df.columns.drop('EmployeeID').tolist()
    else:
        columns_to_use = [col for col in columns_to_use if col != 'EmployeeID'] + mandatory_columns
        columns_to_use = list(set(columns_to_use))  # Remove duplicates, if any

    df = df[columns_to_use]

    # Convert 'Attrition' to binary
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    # Identify binary categorical columns and encode them directly
    binary_columns = ['Gender', 'OverTime']  # Add any other binary columns here
    for col in binary_columns:
        if col in df.columns:
            unique_values = df[col].unique()
            if len(unique_values) == 2:  # Ensure the column is truly binary
                df[col] = df[col].map({unique_values[0]: 0, unique_values[1]: 1})

    # Automatically identify remaining categorical columns for one-hot encoding
    categorical_columns = df.select_dtypes(include=['object']).columns.drop('Attrition', errors='ignore').tolist()

    # Perform one-hot encoding on the remaining categorical columns
    df = pd.get_dummies(df, columns=categorical_columns)

    # Separate features and target
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale numeric features, fitting the scaler on the training set only
    scaler = StandardScaler()
    numeric_columns = X_train.select_dtypes(include=['int64', 'float64']).columns
    X_train[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
    X_test[numeric_columns] = scaler.transform(X_test[numeric_columns])

    return X_train, X_test, y_train, y_test

# You can now call this function, specifying columns_to_use if desired. 'Attrition' will always be included.
