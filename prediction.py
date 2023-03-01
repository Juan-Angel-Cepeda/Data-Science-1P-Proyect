import joblib

def predict(data, model):
    if model == 'Linear Regression':
        model = joblib.load('linear_regression_model.pkl')
    elif model == 'Decision Tree Regression':
        model =joblib.load('tree_regression_model.pkl')
    elif model == 'Random Forest Regression':
        model == joblib.load('random_forest_model.pkl')
    elif model == 'Suport Vector Machine':
        model == joblib.load('svr_regression_model.pkl')


    pipeline =joblib.load("full_pipeline.pkl")
    data = pipeline.transform(data)
    
    return model.predict(data)


