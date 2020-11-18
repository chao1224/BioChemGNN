import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, roc_auc_score, average_precision_score


def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))