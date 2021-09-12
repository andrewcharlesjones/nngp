import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns
from os.path import join as pjoin
from sklearn.gaussian_process.kernels import RBF

import matplotlib

font = {"size": 20}
matplotlib.rc("font", **font)
matplotlib.rcParams["text.usetex"] = True

SAVE_DIR = (
    "/Users/andrewjones/Documents/princeton_webpage/andrewcharlesjones.github.io/assets"
)

lims = [-3, 3]
n = 100
D = 2
p = 1
m = 10

## Generate spatial coordinates
X = np.random.uniform(low=lims[0], high=lims[1], size=(n, D))

## Sort coordinates somehow (here by x value)
sorted_idx = np.argsort(X[:, 0])
X_sorted = X[sorted_idx]

## Form distance matrix
dist_mat = pairwise_distances(X_sorted)

## Get neighbor mask
## Element (i, j) is 1 if x_j is a neighbor of x_i
neighbor_mask = np.zeros((n, n))
for ii in range(n):
	curr_sorted_idx = np.argsort(dist_mat[ii, :])
	curr_sorted_idx = curr_sorted_idx[curr_sorted_idx < ii]
	neighbor_idx = curr_sorted_idx[:min(ii, m)]
	neighbor_mask[ii, :][neighbor_idx] = 1

# np.fill_diagonal(neighbor_mask, 1)

plt.figure(figsize=(5, 5))
sns.heatmap(neighbor_mask, cbar=False)
plt.ylabel(r"$i$")
plt.xlabel(r"$j$")
plt.title("Neighbor mask")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "neighbor_mask_nngp.png"))
# plt.show()
plt.close()

## Form B and F matrices
B = np.zeros((n, n))
F = np.zeros((n, n))
kernel_func = RBF()

for ii in range(n):
	curr_neighbor_idx = np.where(neighbor_mask[ii] == 1)[0]
	curr_Xi = np.expand_dims(X_sorted[ii, :], 0)
	curr_Nxi = X_sorted[curr_neighbor_idx]
	K_xi_Nxi = kernel_func(curr_Xi, curr_Nxi)
	K_xi_xi = kernel_func(curr_Xi, curr_Xi)
	K_Nxi_Nxi = kernel_func(curr_Nxi, curr_Nxi)

	curr_B = K_xi_Nxi @ np.linalg.solve(K_Nxi_Nxi, np.eye(len(curr_neighbor_idx)))
	curr_F = K_xi_xi - curr_B @ K_xi_Nxi.T

	B[ii][curr_neighbor_idx] = curr_B.squeeze()
	F[ii, ii] = curr_F.squeeze()

## Make diagonal elements 1
np.fill_diagonal(B, 1)

plt.figure(figsize=(5, 5))
sns.heatmap(B, cbar=False, center=0)
plt.ylabel(r"$i$")
plt.xlabel(r"$j$")
plt.title(r"$B$")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "B_matrix_nngp.png"))
# plt.show()
plt.close()




precision_mat = B.T @ np.linalg.solve(F, np.eye(n)) @ B
plt.figure(figsize=(14, 7))
plt.subplot(121)
sns.heatmap(kernel_func(X_sorted, X_sorted), center=0, cbar=False)
plt.ylabel(r"$i$")
plt.xlabel(r"$j$")
plt.title(r"$\Sigma$")
plt.tight_layout()

plt.subplot(122)
sns.heatmap(np.linalg.solve(precision_mat, np.eye(n)), center=0, cbar=False)
plt.ylabel(r"$i$")
plt.xlabel(r"$j$")
plt.title(r"$\widetilde{\Sigma}$")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "covariance_matrix_nngp.png"))
plt.close()




## Form full precision matrix
precision_mat = B.T @ np.linalg.solve(F, np.eye(n)) @ B
plt.figure(figsize=(14, 7))
plt.subplot(121)
sns.heatmap(np.linalg.solve(kernel_func(X_sorted, X_sorted), np.eye(n)), center=0, cbar=False)
plt.ylabel(r"$i$")
plt.xlabel(r"$j$")
plt.title(r"$\Sigma^{-1}$")
plt.tight_layout()

plt.subplot(122)
sns.heatmap(precision_mat, center=0, cbar=False)
plt.ylabel(r"$i$")
plt.xlabel(r"$j$")
plt.title(r"$\widetilde{\Sigma}^{-1}$")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "precision_matrix_nngp.png"))
plt.close()
# plt.show()


plt.figure(figsize=(5, 5))
plt.scatter(X[:, 0], X[:, 1], color="black")
plt.xlabel("Spatial coordinate 1")
plt.ylabel("Spatial coordinate 2")
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "x_data_nngp.png"))
plt.close()





plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], color="black")

idx_to_plot = 67
curr_neighbor_idx = np.where(neighbor_mask[idx_to_plot] == 1)[0]
plt.scatter(X_sorted[idx_to_plot, 0], X_sorted[idx_to_plot, 1], color="red", label=r"$x_i$")
plt.scatter(X_sorted[curr_neighbor_idx][:, 0], X_sorted[curr_neighbor_idx][:, 1], color="green", label=r"$N(x_i)$")

plt.xlabel("Spatial coordinate 1")
plt.ylabel("Spatial coordinate 2")
plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.tight_layout()
plt.savefig(pjoin(SAVE_DIR, "x_neighbor_example_nngp.png"))
# plt.close()
plt.show()



import ipdb; ipdb.set_trace()

