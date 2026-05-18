# sample prediction
def predict_sample(model, X_test):
    sample = X_test.iloc[[0]]
    prediction = model.predict(sample)

    print("\nSample Prediction")
    print(prediction[0])
