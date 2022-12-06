from flask import Flask
from flask import request
import numpy as np
import pandas as pd

from workflow import ica_workflow, pca_workflow, preprocess_workflow
app = Flask(__name__)

@app.route('/submit_eeg_ica/<channels>/<iterations>', methods=['GET', 'POST'])
def submit_eeg_ica(channels=14, iterations=50):
    data = np.array(pd.read_csv(request.files['data']))
    if len(data) < 1:
        return {"Array not found":False}

    channels, iterations = int(channels), int(iterations)
    data = ica_workflow.run_workflow(data, channels, iterations)

    return {str(data): True}

@app.route('/submit_eeg_pca/<channels>', methods=['GET', 'POST'])
def submit_eeg_pca(channels=1):
    data = np.array(pd.read_csv(request.files['data']))
    if len(data) < 1:
        return {"Array not found":False}

    data = pca_workflow.run_workflow(data, channels)
    return {str(data):True}


@app.route('/submit_eeg_preprocessing', methods=['GET', 'POST'])
def submit_eeg_pca():
    data = np.array(pd.read_csv(request.files['data']))
    if len(data) < 1:
        return {"Array not found": False}
    data = preprocess_workflow.run_workflow(data)
    return {str(data): True}


if __name__ == '__main__':
    app.run()