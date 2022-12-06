import pandas as pd
import numpy as np
from numpy import genfromtxt

eeg_data = genfromtxt('Dunn2.csv', delimiter=',', dtype=np.float16)
eeg_data_no_blink = eeg_data[:,:14]
print(eeg_data_no_blink)
# np.savetxt("Dunn2_NoBlink.csv", eeg_data_no_blink, delimiter=",")
eeg_data_no_blink.tofile('data2.csv', sep = ',')