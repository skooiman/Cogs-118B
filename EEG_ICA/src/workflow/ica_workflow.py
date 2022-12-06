from src.algorithms import preprocess, pca, ica
from src.workflow import pca_workflow


def run_workflow(data, channels=None, iterations=None):
    if channels is None:
        channels = data.channels
    if iterations is None:
        iterations = 50
    pca_workflow.run_workflow(data, channels)
    return ica.ica(data[:,:channels], iterations, 1e-5)