import pandas as pd
import numpy as np

ground_truth = pd.read_csv('truth.csv')
submission = pd.read_csv('out/result.csv')

print(np.sum(ground_truth['target'] == submission['target']) / submission.shape[0])
