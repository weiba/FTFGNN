import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch.utils.data as Data
from model import *
import utils


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on fMRI data.")
    parser.add_argument('--data-path', type=str, required=True, help="Path to the data file.")
    parser.add_argument('--embed-size', type=int, default=64, help="Embedding size.")
    parser.add_argument('--hidden-size', type=int, default=128, help="Hidden size.")
    parser.add_argument('--lower-dim', type=int, default=8)
    parser.add_argument('--fc-dim', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument('--layer-num', type=int, default=4, help="Number of Layers.")
    parser.add_argument('--lamb', type=float, default=0.5, help="Threshold for Adjacency Matrix Construction.")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size.")
    
    return parser.parse_args()

args = parse_args()
print(args)
task_name = os.path.basename(args.data_path).split('.')[0].replace('_', '-')
if task_name.endswith('-new'):
    task_name = task_name[:-4] 
print(f"Task Name: {task_name}")
np.random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("USING DEVICE ",device)
layer_configs = list(range(1, args.layer_num+1))
results = {
    'ACC': [],
    'SEN': [],
    'SPE': [],
    'BAC': [],
    'F1': [],
    'MCC': [],
    'AUC': [],
    'epoch': []
}
    
print("Loading data......")
roi_features, adj_mats, labels = utils.load_data(args.data_path, mean=args.lamb)
print(roi_features.shape, adj_mats.shape, labels.shape)

train_indices, test_indices = train_test_split(np.arange(len(labels)), test_size=0.2, stratify=labels)
remain_roi_features = roi_features[train_indices]
remain_labels = labels[train_indices]
remain_adjs = adj_mats[train_indices]
test_roi_features = roi_features[test_indices]
test_labels = labels[test_indices]
test_adjs = adj_mats[test_indices]
test_dataset = Data.TensorDataset(test_roi_features, test_adjs, test_labels)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)


layer_num = args.layer_num
print(f"\nTraining with {layer_num} layers......")

ACClist, SENlist, SPElist, BAClist, F1list, MCClist, AUClist,  epoch_list = [], [], [], [], [], [], [], []

for fold, (train_index, val_index) in enumerate(kfold.split(remain_roi_features, remain_labels)):
    print(f"Fold {fold + 1}/{kfold.n_splits}")

    train_ACC, train_SEN, train_SPE, train_BAC, train_F1, train_MCC, train_AUC = 0, 0, 0, 0, 0, 0, 0
    test_ACC, test_SEN, test_SPE, test_BAC, test_F1, test_MCC, test_AUC = 0, 0, 0, 0, 0, 0, 0
    ACC, SEN, SPE, BAC, F1, MCC, AUC = 0, 0, 0, 0, 0, 0, 0

    model = FGN(embed_size=args.embed_size, feature_size=90, seq_length=116, hidden_size=args.hidden_size,
                layer_num=layer_num, fc_dim=args.fc_dim, lower_dim=args.lower_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss().to(device)

    train_roi_features = remain_roi_features[train_index]
    train_labels = remain_labels[train_index]
    train_adj = remain_adjs[train_index]

    val_roi_features = remain_roi_features[val_index]
    val_labels = remain_labels[val_index]
    val_adj = remain_adjs[val_index]

    train_dataset = Data.TensorDataset(train_roi_features, train_adj, train_labels)
    val_dataset = Data.TensorDataset(val_roi_features, val_adj, val_labels)

    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = Data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_acc = 0.0
    best_model_path = f"{fold + 1}-{task_name}-FTF-GNN-best.pth"
    for epoch in range(1, args.epochs + 1):
        loss_total = 0.0
        all_outputs = []
        all_labels = []
        model.train()
        for step, (train_roi_features, train_adjs, train_labels) in enumerate(train_loader):
            train_roi_features = train_roi_features.to(device)
            train_labels = train_labels.to(device)
            train_adj_batch = train_adjs.to(device)
            output = model(train_roi_features, train_adj_batch)
            loss = loss_func(output, train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
            all_outputs.append(output.cpu())  
            all_labels.append(train_labels.cpu())  

        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        model.eval()
        total_test_loss = 0.
        all_val_outputs = []
        all_val_labels = []
        with torch.no_grad():
            for step, (val_roi_features, val_adjs, val_labels) in enumerate(val_loader):
                val_roi_features = val_roi_features.to(device)
                val_labels = val_labels.to(device)
                val_adj_batch = val_adjs.to(device)
                output_val = model(val_roi_features, val_adj_batch)

                loss_val = loss_func(output_val, val_labels)
                total_val_loss = total_test_loss + loss_val.item()
                all_val_outputs.append(output_val.cpu())
                all_val_labels.append(val_labels.cpu())
        
        all_val_outputs = torch.cat(all_val_outputs, dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        val_ACC, val_SEN, val_SPE, val_BAC, val_F1, val_MCC, val_AUC = utils.stastic_indicators(all_val_outputs, all_val_labels)
        print(f'| epoch {epoch:3d} | val acc {val_ACC * 100:.2f} | sen {val_SEN * 100:.2f} | spe {val_SPE * 100:.2f} | auc {val_AUC * 100:.2f} | F1 {val_F1 * 100:.2f} | mcc {val_MCC * 100:.2f}')
        if val_ACC > best_val_acc:
            best_val_acc = val_ACC
            torch.save(model.state_dict(), best_model_path)

    model.eval()
    total_test_loss = 0.0
    all_test_outputs = []
    all_test_labels = []
    with torch.no_grad():
        for step, (test_roi_features, test_adjs, test_labels_batch) in enumerate(test_loader):
            test_roi_features = test_roi_features.to(device)
            test_labels_batch = test_labels_batch.to(device)
            test_adj_batch = test_adjs.to(device)
            output_test = model(test_roi_features, test_adj_batch)

            loss_test = loss_func(output_test, test_labels_batch)
            total_test_loss += loss_test.item()

            all_test_outputs.append(output_test.cpu())
            all_test_labels.append(test_labels_batch.cpu())

        all_test_outputs = torch.cat(all_test_outputs, dim=0)
        all_test_labels = torch.cat(all_test_labels, dim=0)
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

        test_ACC, test_SEN, test_SPE, test_BAC, test_F1, test_MCC, test_AUC = utils.stastic_indicators(all_test_outputs, all_test_labels)
    print(f'Fold {fold + 1} Test | acc {test_ACC * 100:.2f} | sen {test_SEN * 100:.2f} | spe {test_SPE * 100:.2f} | bac {test_BAC * 100:.2f} | F1 {test_F1 * 100:.2f} | mcc {test_MCC * 100:.2f} | auc1 {test_AUC * 100:.2f}')
    ACClist.append(test_ACC)
    SENlist.append(test_SEN)
    SPElist.append(test_SPE)
    BAClist.append(test_BAC)
    F1list.append(test_F1)
    MCClist.append(test_MCC)
    AUClist.append(test_AUC)
    

print(ACClist)
print(SENlist)
print(SPElist)
print(BAClist)
print(F1list)
print(MCClist)
print(AUClist)

results['ACC'].append(sum(ACClist) / len(ACClist))
results['SEN'].append(sum(SENlist) / len(SENlist))
results['SPE'].append(sum(SPElist) / len(SPElist))
results['BAC'].append(sum(BAClist) / len(BAClist))
results['F1'].append(sum(F1list) / len(F1list))
results['MCC'].append(sum(MCClist) / len(MCClist))
results['AUC'].append(sum(AUClist) / len(AUClist))


print(args)
print(results)