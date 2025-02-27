import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import  roc_auc_score
from sklearn.preprocessing import StandardScaler

np.random.seed(123)
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

def load_data(data_path,mean):
  data = np.load(data_path)
  roi_data = data['fmri_data']
  roi_feats = []
  adj_mats = []
  for i in tqdm(range(len(roi_data))):
    roi_feature = roi_data[i]
    a = roi_feature
    scaler = StandardScaler()
    roi_feature = scaler.fit_transform(roi_feature)
    pcc = np.corrcoef(roi_feature)

    adj = np.where(pcc > mean, 1, 0)
    roi_feats.append(roi_feature)
    row, col = np.diag_indices_from(adj)
    adj[row, col] = 1

    adj = normalize_adj(adj)
    adj_mats.append(adj)
  roi_features = np.array(roi_feats)
  roi_features = torch.from_numpy(roi_features).float()
  adj_mats = np.array(adj_mats)
  adj_mats = torch.from_numpy(adj_mats).float()
  if "AD_NC_new.npz" in data_path:
    labels = torch.LongTensor([1] * 222 + [0] * 213)
  elif "LMCI_NC_new.npz" in data_path:
    labels = torch.LongTensor([1] * 192 + [0] * 213)
  elif "LMCI_AD_new.npz" in data_path:
    labels = torch.LongTensor([1] * 192 + [0] * 222)
  elif "EMCI_LMCI_new.npz" in data_path:
    labels = torch.LongTensor([1] * 190 + [0] * 192)
  elif "EMCI_LMCI_ADNI3.npz" in data_path:
    labels = torch.LongTensor([1] * 65 + [0] * 78)
  elif "LMCI_AD_ADNI3.npz" in data_path:
    labels = torch.LongTensor([1] * 78 + [0] * 134)
  elif "EMCI_LMCI_test.npz" in data_path:
    labels = torch.LongTensor([1] * 125 + [0] * 114)     
  elif "LMCI_AD_test.npz" in data_path:
    labels = torch.LongTensor([1] * 114 + [0] * 87)  
  else:
    raise ValueError("Unknown dataset!")
  return roi_features,adj_mats,labels
  
def stastic_indicators(output, labels):
    epsilon = 1e-7  
    predictions = output.max(1)[1]
    
    TP = ((predictions == 1) & (labels == 1)).sum()
    TN = ((predictions == 0) & (labels == 0)).sum()
    FN = ((predictions == 0) & (labels == 1)).sum()
    FP = ((predictions == 1) & (labels == 0)).sum()

    ACC = (TP + TN) / (TP + TN + FP + FN + epsilon)
    SEN = TP / (TP + FN + epsilon)
    P = TP / (TP + FP + epsilon)
    SPE = TN / (FP + TN + epsilon)
    BAC = (SEN + SPE) / 2
    F1 = (2 * P * SEN) / (P + SEN + epsilon)
    MCC = ((TP * TN) - (FP * FN)) / (torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + epsilon)

    labels = labels.cpu().numpy()
    predictions = predictions.cpu().numpy()
    try:
        AUC = roc_auc_score(labels, output[:,1])
    except ValueError:
        AUC = 0.0    

    return ACC, SEN, SPE, BAC, F1, MCC, AUC




