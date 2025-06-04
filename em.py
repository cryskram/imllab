import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

X1 = np.random.normal(2, 1, 200)
X2 = np.random.normal(-1, 0.8, 600)

X = np.concatenate([X1, X2]).reshape(-1, 1)

gmm = GaussianMixture(n_components=2)
gmm.fit(X)

x_grid = np.linspace(X.min(), X.max(), 1000).reshape(-1, 1)
density = np.exp(gmm.score_samples(x_grid))

sns.kdeplot(X.flatten(), label="Actual Density")
plt.plot(x_grid, density, label="GMM")
plt.legend()
plt.show()
