# Brut-v1.1

This repository contains an updated version of Brut (https://github.com/ChrisBeaumont/brut) used in the paper

Assessing the Performance of a Machine Learning Algorithm in Identifying Bubbles in Dust Emission, ApJ in press ([arXiv link](https://arxiv.org/abs/1711.03480))* 

We make slight changes on the modules that Brut import. The current version Brut can be successfully run with the following libraries

* astropy '2.0.2'
* h5py '2.7.0'
* matplotlib '2.0.2'
* numpy '1.13.3'
* scipy '1.0.0'
* skimage '0.13.0'
* sklearn '0.19.1'
* cloud '2.8.5'

We update the retrained model in models/ directory.


## Organization

### models/
Contains the original training model and the model retrained on synthetic images and the orignial traning set.
The synthetic bubble images can be found here (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OSMNDG).

