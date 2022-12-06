import numpy as np

def g(x):
    return np.tanh(x)

# g'(x) = 1-tanh^2(x)
def g_der(x):
    return 1 - np.tanh(x) ** 2


def scale(X):
    # For decorrelated mixture signals projected to PCA space as U = VD
    # Z =(λ^(-1/2))U = (λ^(-1/2))VD
    pass


def recalculate_w(w, X):
    w_new = (X * g(w.T @ X)).mean(axis=1) - g_der(w.T @ X).mean() * w
    w_new /= np.sqrt((w_new ** 2).sum())
    return w_new


def ica(X, iterations, tolerance):
    X = X.T
    components_count = X.shape[0]
    W = np.zeros((components_count, components_count), dtype=X.dtype)  # Initialize empty weights matrix

    for i in range(components_count):
        w = np.random.rand(components_count)
        j = 0
        converged = False
        while j < iterations and not converged:
            w_new = recalculate_w(w, X)

            if i >= 1:
                w_new -= (w_new @ W[:i].T) @ W[:i]

            distance = np.abs(np.abs((w * w_new).sum()) - 1)
            w = w_new

            if distance < tolerance:
                converged = True

            j += 1

        W[i, :] = w

    S = W @ X

    return S
