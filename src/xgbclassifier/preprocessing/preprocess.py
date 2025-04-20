import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

def preprocess(data):

    #define model features
    features = [
        'year', 'event_key', 'comp_level', 'match_number',
        'red_1_win_rate', 'red_2_win_rate', 'red_3_win_rate', 'blue_1_win_rate',
        'blue_2_win_rate', 'blue_3_win_rate', 'red_1_epa_trend', 'red_2_epa_trend',
        'red_3_epa_trend', 'blue_1_epa_trend', 'blue_2_epa_trend', 'blue_3_epa_trend',
        'epa_diff','epa_ratio'
    ]

    #Create target data
    data['winner'] = (data['score_dif'] > 0).astype(int)
    target = "winner"

    #Designate features and target
    X = data[features]
    y = data[target]

    X = pd.get_dummies(X, columns=['event_key', 'comp_level'], drop_first=True)

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Scale characteristics
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Save column names and reconstruct dataframe
    column_names = X.columns

    X_transformed = pd.DataFrame(X_scaled, columns=column_names)

    dir = "src/xgbclassifier/preprocessing"
    
    #Save imputer
    joblib.dump(imputer, f"{dir}/imputer.joblib")

    #Save scaler
    joblib.dump(scaler, f"{dir}/scaler.joblib")

    #Save column names
    joblib.dump(column_names, f"{dir}/column_names.joblib")

    return train_test_split(X_transformed, y, test_size=0.2, random_state=42)

    