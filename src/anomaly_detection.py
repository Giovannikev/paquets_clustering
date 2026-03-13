import numpy as np

def detect_anomalies_dbscan(labels):
    # Dans DBSCAN, les anomalies ont le label -1
    anomalies = np.where(labels == -1)[0]

    return anomalies