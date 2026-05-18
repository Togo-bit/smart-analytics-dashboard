# evaluation
def evaluate_model(
        model,
        problem_type,
        X_test,
        y_test
):
    y_pred = model.predict(X_test)

    # Regression
    if problem_type == 'regression':
        r2 = r2_score(y_test, y_pred)

        mae = mean_absolute_error(y_test, y_pred)

        print("\nRegression results")
        print("-"*40)

        print(f"R2 Score: {r2:.4f}")
        print(f"MAE: {mae:.4f}")
    else:
        accuracy = accuracy_score(y_test, y_pred)

        print("\nClassification results")
        print("-"*40)

        print(f"Accuracy: {accuracy:.4f}")

        print("\nClassification report\n")
        print(classification_report(y_test, y_pred, zero_division=0))
