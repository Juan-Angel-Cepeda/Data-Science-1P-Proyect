import joblib

def predict(data, model):
    if model == 'Linear Regression':
        modelpkl = joblib.load('linear_regression_model.pkl')
    
    elif model == 'Decision Tree Regression':
        modelpkl =joblib.load('tree_regression_model.pkl')
    
    elif model == 'Random Forest Regression':
        modelpkl == joblib.load('random_forest_model.pkl')
    
    elif model == 'Support Vector Regression':
        modelpkl == joblib.load('svr_regression_model.pkl')

    pipeline =joblib.load("full_pipeline.pkl")
    data = pipeline.transform(data)
    
    return modelpkl.predict(data)


