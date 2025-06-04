import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

transactions = [
    ["bread", "milk"],
    ["bread", "diaper", "beer", "egg"],
    ["milk", "diaper", "beer", "cola"],
    ["bread", "milk", "diaper", "beer"],
    ["bread", "milk", "diaper", "cola"],
]

te = TransactionEncoder()
te_ary = te.fit_transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

frequent = apriori(df, min_support=0.6, use_colnames=True)
rules = association_rules(frequent, metric="lift", min_threshold=1)

print("Frequent list: ", frequent)
print("Rules: ", rules[["antecedents", "consequents", "support", "confidence", "lift"]])
