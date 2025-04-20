import pandas as pd
import joblib

def predict(data):
    #import model, scaler, imputer and column names
    dir = "src/xgbclassifier"

    model = joblib.load(f"{dir}/model/xgbclassifier.joblib")

    imputer = joblib.load(f"{dir}/preprocessing/imputer.joblib")
    scaler = joblib.load(f"{dir}/preprocessing/scaler.joblib")
    column_names = joblib.load(f"{dir}/preprocessing/column_names.joblib")

    # Process data
    X = data[[
        'year', 'event_key', 'comp_level', 'match_number',
        'red_1_win_rate', 'red_2_win_rate', 'red_3_win_rate', 'blue_1_win_rate',
        'blue_2_win_rate', 'blue_3_win_rate', 'red_1_epa_trend', 'red_2_epa_trend',
        'red_3_epa_trend', 'blue_1_epa_trend', 'blue_2_epa_trend', 'blue_3_epa_trend',
        'epa_diff','epa_ratio'
    ]]
    X = pd.get_dummies(X, columns=['event_key', 'comp_level'], drop_first=True)

    # Reindex to match the training column order
    X = X.reindex(columns=column_names, fill_value=0)

    # Imput characteristics
    X_imputed = imputer.transform(X)

    # Scale Characteristics
    X_scaled = scaler.transform(X_imputed)

    # Make predictions
    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)

    # Turn actual match winner to binary value
    actual = data["winning_alliance"].map({"red": 1, "blue": 0})    

    print(f"{sum(preds == actual)}/{len(data)} ({sum(preds == actual)/len(data)*100}%)")
    
    predictions =  [(k, float(p)) for k, p in zip(data["match_key"], probs,)]
    return predictions


