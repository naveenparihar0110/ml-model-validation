import pytest
from sklearn.metrics import accuracy_score
from model_pipeline import load_data, train_model


@pytest.mark.xfail(
    reason="Known gender bias in Adult Income dataset (baseline model)"
)
def test_gender_bias():
    X_train, X_test, y_train, y_test, df_test = load_data()
    model = train_model(X_train, y_train)

    male_idx = df_test["sex_Male"] == 1
    female_idx = df_test["sex_Male"] == 0

    acc_male = accuracy_score(y_test[male_idx], model.predict(X_test[male_idx]))
    acc_female = accuracy_score(y_test[female_idx], model.predict(X_test[female_idx]))

    assert abs(acc_male - acc_female) < 0.15
