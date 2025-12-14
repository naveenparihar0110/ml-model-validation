from model_pipeline import load_data, train_model
from sklearn.metrics import accuracy_score

def test_overfitting():
    X_train, X_test, y_train, y_test, _ = load_data()
    model = train_model(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    assert (train_acc - test_acc) < 0.05
