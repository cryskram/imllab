import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_acc, test_acc = [], []

for i in range(1, 10):
    knn = KNeighborsClassifier(i)
    knn.fit(X_train, y_train)
    train_acc.append(knn.score(X_train, y_train))
    test_acc.append(knn.score(X_test, y_test))


plt.plot(range(1, 10), train_acc, "s-", label="Train Accuracy")
plt.plot(range(1, 10), test_acc, "o-", label="Test Accuracy")
plt.xlabel("i")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
