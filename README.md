# Connectomics Thesis

This repository contains the code used in the thesis titled "Machine Learning Prediction of Cognition Through Structure-Function Coupling in the Frontoparietal and Default Mode Networks". 
It focuses on the estimation of structural and functional connectivity between the DMN and FPN, the training and testing of a neural network model, and the the training and testing of a lasso CV model.

## Repository Structure

Here's a brief explanation of the files included in this repository:

1. **structural_networks.ipynb**: Constructs the structural connectivity matrices.
2. **functional_networks.ipynb**: Constructs the functional connectivity matrices.
3. **npy_to_mat.ipynb**: Combines the structural and functional matrices into a .mat file. It also aggregates the upper triangles of the matrices and splits the data into training and test .mat files.
4. **network.py**: Provides the framework for the neural network.
5. **train.py**: Used for training the neural network.
6. **reload.py**: Used for making predictions with the pre-trained network.
7. **nn_pearson_corr.ipynb**: Calculates the correlation between empirical and predicted functional matrices to assess the performance of the neural network.
8. **residuals.ipynb**: Regresses the structural connectivity out of the empirical and predicted functional matrices.
9. **lasso_regression.ipynb**: Trains a LassoCV model, makes predictions, and evaluates the performance of the model.
10. **plots.ipynb**: Generates the plots used in the thesis report.

All files are intended to be run in the order they are listed above.

## References and Acknowledgements

* The **structural_networks.ipynb** and **functional_networks.ipynb** were adapted from code provided by Lars Smolders from the Brain Tumour and Cognition group at the Elisabeth-TweeSteden Ziekenhuis. The original two python scripts he provided can be found in the 'lars' folder in this repository.

* The **network.py**, **train.py**, and **reload.py** were adapted from code provided by Dr. Tabinda Sarwar from the RMIT School of Computing Technologies. The original code can be found at this [Github repository](https://github.com/sarwart/mapping_SC_FC).

I am grateful to both Lars Smolders and Dr. Tabinda Sarwar for their contributions to this work.
