import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import precision_recall_fscore_support

# calculate the fpr and tpr for all thresholds of the classification
y_test = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
preds = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'me'])

print(precision_recall_fscore_support(y_test, preds, average='macro'))
