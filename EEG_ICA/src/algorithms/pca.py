import numpy as np
from sklearn.preprocessing import normalize

def eigsort(V, eigvals):
    # Sort the eigenvalues from largest to smallest. Store the sorted
    # eigenvalues in the column vector lambd.
    lohival = np.sort(eigvals)
    lohiindex = np.argsort(eigvals)
    lambd = np.flip(lohival)
    index = np.flip(lohiindex)
    Dsort = np.diag(lambd)

    # Sort eigenvectors to correspond to the ordered eigenvalues. Store sorted
    # eigenvectors as columns of the matrix vsort.
    M = np.size(lambd)
    Vsort = np.zeros((M, M))
    for i in range(M):
        Vsort[:, i] = V[:, index[i]]
    return Vsort, Dsort

def PCA(zeroed_data, channels):
    D, V = np.linalg.eig(zeroed_data.transpose() @ zeroed_data)
    V, D = eigsort(V, D)
    U = zeroed_data @ V
    U = normalize(U, norm='l2', axis=0)
    reconstructed_dataset = U.T @ zeroed_data[:, :]
    reconstructed = U[:, :] @ np.array((reconstructed_dataset[:][:]))
    #     reconstructed = U[:,:10]@np.array((reconstructed_dataset[:10][:]))
    if channels < len(U[0]):
        channels = len(U[0])
    return U[:,:channels]