from sklearn.ensemble import RandomForestRegressor

def train_rf_model(train, params, log):
    X = train.drop(columns=[params.id_col, params.time_col, params.dependent_var, "target"])
    y = train["target"]
    rf = RandomForestRegressor()
    rf.fit(X, y)
    return rf