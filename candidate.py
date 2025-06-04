import pandas as pd

data = pd.read_csv("dataset.csv")
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values


def candidate_elimination(concepts, target):
    s = concepts[0].copy()
    g = ["?" * len(s)]

    for i, h in enumerate(concepts):
        if target[i].lower() == "yes":
            s = ["?" if s[j] != h[j] else s[j] for j in range(len(s))]
        else:
            g = [
                gen
                for gen in g
                if all(gen[j] == "?" or gen[j] != h[j] for j in range(len(s)))
            ]

    return [s], g


print(*candidate_elimination(concepts, target), sep="\n")
