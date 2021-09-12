import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns
from os.path import join as pjoin
from sklearn.gaussian_process.kernels import RBF
from scipy.stats import multivariate_normal as mvn

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

SAVE_DIR = (
    "/Users/andrewjones/Documents/princeton_webpage/andrewcharlesjones.github.io/assets"
)

lims = [-10, 10]
n = 40
ntest = 500
D = 1
p = 1
m = 10
noise_variance_true = 0.1

kernel_func = RBF()


## Generate spatial coordinates
X = np.random.uniform(low=lims[0], high=lims[1], size=(n, D))
# Xtest = np.random.uniform(low=lims[0], high=lims[1], size=(ntest, D))
Xtest = np.expand_dims(np.linspace(lims[0], lims[1], ntest), 1)
X_full = np.concatenate([X, Xtest], axis=0)

K_XX_true = kernel_func(X_full, X_full)
Y_full = mvn.rvs(np.zeros(n + ntest), K_XX_true) + np.random.normal(size=n + ntest, scale=np.sqrt(noise_variance_true))
Y = Y_full[:n]
Ytest = Y_full[n:]

## Sort coordinates somehow (here by x value)
sorted_idx = np.argsort(X[:, 0])
X_sorted = X[sorted_idx]
Y_sorted = Y[sorted_idx]
sorted_idx = np.argsort(Xtest[:, 0])
Xtest_sorted = Xtest[sorted_idx]
Ytest_sorted = Ytest[sorted_idx]

## Form distance matrix
dist_mat = pairwise_distances(Xtest_sorted, X_sorted)



## Get neighbor mask
## Element (i, j) is 1 if x_j is a neighbor of x_i
neighbor_mask = np.zeros((ntest, n))
for ii in range(ntest):
	curr_sorted_idx = np.argsort(dist_mat[ii, :])
	# curr_sorted_idx = curr_sorted_idx[curr_sorted_idx < ii]
	# neighbor_idx = curr_sorted_idx[:min(ii, m)]
	neighbor_idx = curr_sorted_idx[:m]
	neighbor_mask[ii, :][neighbor_idx] = 1


preds = np.zeros(ntest)
for ii in range(ntest):

	curr_neighbor_idx = np.where(neighbor_mask[ii] == 1)[0]
	curr_Xtest = np.expand_dims(Xtest_sorted[ii, :], 0)
	curr_Nxi = X_sorted[curr_neighbor_idx]
	curr_Y = Y_sorted[curr_neighbor_idx]
	K_xi_Nxi = kernel_func(curr_Xtest, curr_Nxi)
	K_Nxi_Nxi = kernel_func(curr_Nxi, curr_Nxi)

	K_Nxi_Nxi_inv = np.linalg.solve(K_Nxi_Nxi + noise_variance_true * np.eye(len(curr_neighbor_idx)), np.eye(len(curr_neighbor_idx)))
	curr_B = K_xi_Nxi @ K_Nxi_Nxi_inv
	curr_pred = np.dot(curr_B, curr_Y)
	preds[ii] = curr_pred


# Get vanilla GP predictions
K_XstarX = kernel_func(Xtest_sorted, X_sorted)
K_XX = kernel_func(X_sorted, X_sorted)
K_XX_inv = np.linalg.solve(K_XX + noise_variance_true * np.eye(n), np.eye(n))
preds_vanilla = K_XstarX @ K_XX_inv @ Y_sorted

plt.figure(figsize=(10, 5))
plt.scatter(X_sorted, Y_sorted, color="black", label="Data")
plt.plot(Xtest_sorted, preds, color="red", label="Predictions, NNGP")
plt.plot(Xtest_sorted, preds_vanilla, color="green", label="Predictions, GP")
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "nngp_predictions.png"))
plt.show()

# plt.subplot(121)
# plt.scatter(Ytest_sorted, preds)
# plt.subplot(122)
# plt.scatter(Ytest_sorted, preds_vanilla)
# plt.show()

import ipdb; ipdb.set_trace()

