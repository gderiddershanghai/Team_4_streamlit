import pandas as pd
from ml_logic.preprocessor import preprocess_and_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

def upsample_data(X_train=None, y_train=None, method='none', random_state=1):
    """
    Applies various sampling techniques to balance the dataset or loads a CGAN-generated dataset.

    Parameters:
    - X_train (pd.DataFrame, optional): Features of the training data. If None, data will be loaded via preprocess_and_split.
    - y_train (pd.Series or np.ndarray, optional): Target variable of the training data. If None, data will be loaded via preprocess_and_split.
    - method (str, optional): The sampling method to use. Options are 'none' (default), 'undersample', 'oversample', 'smote', and 'cgan'.
    - random_state (int, optional): The random state to use for reproducible results. Defaults to 1.

    Returns:
    - pd.DataFrame: The resampled or CGAN-generated feature set.
    - pd.Series or np.ndarray: The resampled or CGAN-generated target variable.

    Raises:
    - ValueError: If an unrecognized method is provided.
    """
    # Load data using the preprocess_and_split function if not provided
    if X_train is None or y_train is None:
        X_train, _, y_train, _ = preprocess_and_split()

    # No resampling, return original data
    if method == 'none':
        return X_train, y_train
    # Undersampling
    elif method == 'undersample':
        sampler = RandomUnderSampler(random_state=random_state)
    # Oversampling
    elif method == 'oversample':
        sampler = RandomOverSampler(random_state=random_state)
    # Synthetic Minority Over-sampling Technique
    elif method == 'smote':
        sampler = SMOTE(random_state=random_state)
    # Conditional GAN for synthetic data generation
    elif method == 'cgan':
        cgan_fp = '../Data/gan_generated_df.csv'  # File path to CGAN-generated data
        cgan_df = pd.read_csv(cgan_fp)
        cols = X_train.columns
        # Split CGAN data into features and target variable based on the original data columns
        X_train_resampled, y_train_resampled = cgan_df[cols], cgan_df['Attrition']
        return X_train_resampled, y_train_resampled
    else:
        raise ValueError("Method not recognized. Choose from 'none', 'undersample', 'oversample', 'smote', or 'cgan'.")

    # Apply chosen resampling method
    if method in ['undersample', 'oversample', 'smote']:
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)

    return X_train_resampled, y_train_resampled
