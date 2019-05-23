# Prediction of Intramolecular Reorganization Energy Using Machine Learning
## Introduction


We predict reorganization energy of organic semiconductors using deep neural networks. We experimented with Circular Fingerprints(CF), Molecular Signatures, and Molecular Transform Descriptors. The code can be used with other descriptors as well, since the descriptors are not generated within the code but read from the input files.


## Required Packages
- TensorFlow
- Keras
- scikit-learn

## How to run the scripts

- Edit settings.ini and settings.py with the proper input and output filenames. 
- In gridSearchOsc.py the hyperparameters need to be set.
- Then run 

```
python gridSearchOsc.py
```
This will run all the folds and output statistics into the given outputfile.

