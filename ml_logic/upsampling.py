import pandas as pd
from ml_logic.preprocessor import preprocess_and_split

def upsample_data(X_train=None, y_train=None, method='none', random_state=1):
    if X_train is None or y_train is None:
        X_train, _, y_train, _ = preprocess_and_split()

    if method == 'none':
        return X_train, y_train
    elif method == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=random_state)
    elif method == 'oversample':
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=random_state)
    elif method == 'smote':
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=random_state)
    elif method == 'cgan':
        cgan_fp = 'ml_logic/gan_generated_df.csv'
        cgan_df = pd.read_csv(cgan_fp)
        cols = X_train.columns
        return cgan_df[cols], cgan_df['Attrition']
    else:
        raise ValueError("Choose from 'none', 'undersample', 'oversample', 'smote', or 'cgan'")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res
