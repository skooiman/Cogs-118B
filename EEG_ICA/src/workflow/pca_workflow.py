from src.algorithms import preprocess, pca

def run_workflow(data, channels):
    data = preprocess.detect_outliers(data)
    data = preprocess.remove_drift(data)
    data = preprocess.zero_data(data)
    return pca.PCA(data, channels)
