# Brut

This repository contains the code and manuscript text used in the paper

*The Milky Way Project: Leveraging Citizen Science and Machine Learning to Detect Interstellar Bubbles. Beaumont, Goodman, Kendrew, Williams, Simpson 2014, ApJS, in press*

The `v1` tag represents the state of the code at the time of publication.


## Organization

### bubbly/
Contains the python library used to fit Random Forest classification models to Spitzer images

### figures/
Contains code to generate figures in the paper

### notebooks/
Contains several IPython notebooks in various states of organization -- some are polished documents describing aspects of the analysis, others are temporary workbooks.

### paper/
Contains the manuscript text itself

### scripts/
Python scripts to fit models and generate other derived data products


## Reproduction

This repository is MIT Licensed.

To reproduce the figures and models generated for the paper, type:

```
python setup.py develop
cd bubbly/data && make
cd paper
make
```

Though I promise you you'll have to play with dependencies to get this all set up :)

## Dependencies

Brut is built on top of several python libraries, and uses data from the GLIMPSE and MIPSGAL surveys from the Spitzer Space Telescope. You'll need the following libraries

* aplpy
* astropy
* h5py
* IPython
* matplotlib
* numpy
* scipy
* skimage
* sklearn
* picloud
* WiseRF

In addition, you need to download the GLIMPSE and MIPSGAL mosaic data. The Makefile inside bubbly/data does this.
