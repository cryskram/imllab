import pandas as pd

data = pd.read_csv("dataset.csv")
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values


def find_s(concepts, target):
    hypo = concepts[0].copy()
    for i in range(1, len(concepts)):
        if target[i].lower() == "yes":
            for j in range(len(hypo)):
                if hypo[j] != concepts[i][j]:
                    hypo[j] = "?"

    return hypo


result = find_s(concepts, target)
print("Final hypothesis: ", result)
