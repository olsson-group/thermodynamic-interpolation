# Thermodynamic Interpolation

This repository contains an implementation of methods and experiments presented in the paper [Thermodynamic Interpolation: A generative approach to molecular thermodynamics and kinetics](https://pubmed.ncbi.nlm.nih.gov/39988824/).

![](toc.png)

## Installation
Follow the below steps to set up environments and install the required dependencies.

### Cloning the repository
Run the following in a terminal.
```
git clone git@github.com:olsson-group/thermodynamic-interpolation.git
cd thermodynamic-interpolants
```

### Installing dependencies
We have provided two installation files to reproduce our Conda environments. To install an environments for general use throughout this repository, run the following
```
conda env create -f ti_env.yml
```

In some cases the general purpose environment is not sufficent. To avoid package incompatibilities, we used a second environment when evaluating energies. To reproduce this environment, run the following
```
conda env create -f ti_energy_env.yml
```

## General Usage
This repository contains experiments for both the lower-dimesnional double well system and molecular data. Our training data can be found [here](https://drive.google.com/file/d/1PfJcwlIJ5VKoIt4oCIxKe9yTxgOTSm23/view?usp=sharing).

### Asymmetric double well
The folder "adw/" contains code for experiments related to the Asymmetric Double Well potential. 

### Molecular data
The folder "mdqm9/" contains code for experiments related to the molecular data.


## Citations
To cite this work, please use the following
```
TBD...
```


