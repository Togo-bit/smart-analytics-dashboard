# train data
def train_data(
        model_name,
        problem_type,
        X_train,
        y_train
):
    # Regression
    if problem_type == "regression":
        if model_name == "linear_regression":
            model = LinearRegression()
        elif model_name == "decision_tree":
            model = DecisionTreeRegressor(random_state=42)
        else:
            raise ValueError('Invalid regression model')

    # Classification
    else:
        if model_name == 'logistic_regression':
            model = LogisticRegression(max_iter=1000)
        elif model_name == 'decision_tree':
            model = DecisionTreeClassifier(random_state=42)
        else:
            raise ValueError('Invalide classification model')

    model.fit(X_train, y_train)
    return model
