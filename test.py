import numpy as np

confusion_matrix = np.array(
    [[9, 3, 2],
     [0, 6, 1],
     [1, 1, 7]])

# 2-TP/TN/FP/FN的计算
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.sum() - (FP + FN + TP)
FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)
print(TP)
print(TN)
print(FP)
print(FN)
