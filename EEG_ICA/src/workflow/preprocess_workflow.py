from src.algorithms import preprocess


def run_workflow(data):
    data = preprocess.detect_outliers(data)
    data = preprocess.remove_drift(data)
    return preprocess.zero_data(data)
