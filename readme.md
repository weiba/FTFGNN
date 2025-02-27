# Frequency-Time Fusion Graph Neural Network (FTF-GNN)

Source code and data for "Integrating Time and Frequency Domain Features of fMRI Time Series for Alzheimer’s Disease Classification Using Graph Neural Networks"



## Introduction

FTF-GNN：A model integrates frequency - and time - domain features for robust AD classification, considering both asynchronous and synchronous brain-region interaction.![](D:\desktop\3111.jpg)

## Requirements

- pytorch==1.12.1+cu113

- numpy==1.23.5

- scipy==1.10.1

- scikit-learn==1.2.2

- tqdm==4.65.0

- positional_encodings==6.0.1 

  You can install the required dependencies by running:

  ```python
  pip install -r requirements.txt
  ```

  In addition, CUDA 11.3 have been used on NVIDIA GeForce RTX 3090.

  

## File Description

### Main Code Files

- **model.py**: Contains the core implementation of the FTF-GNN model. The model integrates both time-domain and frequency-domain features for classification tasks.

- **Positional_encoding.py**: Defines the learnable positional encoding used to enhance the time-domain features. 

- **train-test.py**: This script is responsible for training, validating, and testing the model. It accepts the dataset path, hyperparameters, and other configurations as input arguments. The model is trained on different fMRI classification tasks and evaluated based on the validation and test data. The best model based on validation performance is saved during the process.

- **utils.py**: This file contains utility functions used throughout the project. These include functions for data loading, calculating evaluation metrics (e.g., accuracy, sensitivity, specificity, AUC), computing adjacency matrices, normalizing adjacency matrices, and other preprocessing tasks for the fMRI data.
- **idp-test.py**: This script is used for independent testing on two tasks. It leverages AD, EMCI, and LMCI data from the ADNI3, ADNIDOD, and ADNI-GO datasets as training sets for five-fold cross-validation. Corresponding data from ADNI2 is used as the independent test set .

## Data

Due to privacy and ethical considerations and ADNI's data usage agreement, users need to apply at ADNI and download these data(https://adni.loni.usc.edu/).We provide preprocessed fMRI data files in the `data` directory. The data files include:

- **AD Diagnosis (NC vs. AD)**: `AD_NC_new.npz`
- **Early MCI Diagnosis (NC vs LMCI)**: `LMCI_NC_new.npz`
- **AD and Early Subtype Classification (LMCI vs AD)**: `LMCI_AD_new.npz`
- **Two MCI Subtype Classification (EMCI vs LMCI)**: `EMCI_LMCI_new.npz`

Additionally, we also provide data for independent testing located in the `data/Independent_data_center` directory. The dataset can be loaded using the `load_data` function provided in the `utils.py` file.

Example to load data:

```python
roi_features, adj_mats, labels = utils.load_data('path_to_data/file.npz')
```



### fMRI preprocessing:

DPARSF(http://rfmri.org/DPARSF)



## Usage

### Training for Classification Tasks

You can run the model for classification tasks with the following command. This example shows how to train the model for the **AD vs NC** classification task:

```python
python3 train-test.py --data-path ../data/AD_NC_new.npz --lower-dim 12 --batch-size 64 --fc-dim 512 --hidden-size 90 --embed-size 64 --lr 1e-4 --layer-num 5 --lamb 0.5
```

### Arguments for `train-test.py`:

- `--data-path`: Path to the dataset (must be in `.npz` format).
- `--lower-dim`: Dimension of the projection  for frequency feature. Default is `12`.
- `--batch-size`: Batch size for training. Default is `64`.
- `--fc-dim`: Dimension of the fully connected layers. Default is `512`.
- `--hidden-size`: Size of the hidden layers. Default is `90`.
- `--embed-size`: Size of the node embeddings in hypervariate brain network. Default is `64`.
- `--lr`: Learning rate. Default is `1e-4`.
- `--layer-num`: Number of layers in FourierGNN. Default is `5`.
- `--lamb`: Hyperparameter used in data processing. Default is `0.5`.

### Independent Data Center Testing

For independent data center testing, use the following command to evaluate the model on a separate testing dataset. This example shows how to perform independent testing with the **LMCI vs AD** dataset:

```python
python3 idp-test.py --train-data-path ../data/Independent_data_center/train_data/LMCI_AD_ADNI3.npz --test-data-path ../data/Independent_data_center/test_data/LMCI_AD_test.npz --lower-dim 12 --batch-size 64 --fc-dim 512 --hidden-size 90 --embed-size 64 --layer-num 4 --lamb 0.5
```

### Arguments for `idp-test.py`:

- `--train-data-path`: Path to the training dataset for independent testing.
- `--test-data-path`: Path to the testing dataset for independent testing.
- `--lower-dim`: Dimension of the projection  for frequency feature. Default is `12`.
- `--batch-size`: Batch size for testing. Default is `64`.
- `--fc-dim`: Dimension of the fully connected layers. Default is `512`.
- `--hidden-size`: Size of the hidden layers. Default is `90`.
- `--embed-size`: Size of the node embeddings. Default is `64`.
- `--layer-num`: Number of layers in FourierGNN. Default is 4.
- `--lamb`: Hyperparameter used in data processing. Default is `0.5`.



## Contact



If you have any question regard our code or data, please do not hesitate to open a issue or directly contact me ([weipeng1980@gmail.com](mailto:weipeng1980@gmail.com)).

