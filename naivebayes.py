import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report

df = pd.DataFrame(
    {
        "Feature1": [1, 2, 2, 1, 3, 3, 1, 2, 3, 3],
        "Feature2": ["A", "B", "B", "A", "C", "C", "A", "B", "C", "C"],
        "Class": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
    }
)

df = pd.get_dummies(df, columns=["Feature2"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    df.drop("Class", axis=1), df["Class"], test_size=0.2
)

model = CategoricalNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy: ", acc)
print("Classification report: ", cr)
