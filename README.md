# CFDNet and SURFNet
By: Octavi Obiols-Sales, Abhinav Vishnu, Nicholas Malaya, and [Aparna Chandramowlishwaran](https://hpcforge.eng.uci.edu/).

CFDNet is a deep learning-based accelerator for 2D, coarse-grid, steady-state simulations. SURFNet is an extension of CFDNet, for super-resolution of turbulent flows with transfer learning. SURFNet accelerates fine-grid problems while keeping most data collection at coarse grids.

### Why you might be interested in this repository
In this README file, you will find: 
1. An overview on CFDNet and SURFNet.
2. The dataset used to train and valdiate the _coarse model_ in the SURFNet paper and the command to train and reproduce the training results.
3. Use the already trained models and reproduce the performance results in Table III of the paper. 

Please note that this project is dependent of the physics solver (simpleFoam solver in OpenFOAM). For Step (3), one needs to have OpenFOAM v8 installed. Please also note that the performance results in the paper were obtained when OpenFOAM v8 was under [development](https://openfoam.org/download/8-linux/). Hence, the experiments were conducted using the [Singularity](https://develop.openfoam.com/Development/openfoam/-/issues/1483) image for OpenFOAM v8. New OpenFOAM releases/older OpenFOAM versions/OpenFOAM docker images will not reproduce to the exact same number of iterations and speedup results. However, differences should not be significant and speedups of the same order of magnitude should be observed.

### CFDNet: accelerating steady-state simulations
CFDNet was published at the International Conference in Supercomputing (2020), and the paper can be found [here](https://dl.acm.org/doi/pdf/10.1145/3392717.3392772?casa_token=2Vx83VWZAWwAAAAA:BauwuqoOjxXcjrpfsI1MwemUxyTb3rIfdLnf1zkUX66YCtUmdUNYWJjqf0TPYAIPDhDRX0YhwQ_0).

![cfdnet](https://user-images.githubusercontent.com/58092961/110775315-a4743200-8213-11eb-9c9a-32fe2c9b4c42.jpg)

We summarize the idea as follows:
1. We start the steady-state simulation as usual, with a user-given initial condition, and let the solver carry K iterations -- K _warmup_ iterations, as we call it. Once we reach this intermidiate state, we create our input representation using the flow variable values at that K iteration.
2. We create the input representation described in the publication and pass it to the CNN. The CNN then infers the steady-state -- which is unconstrained (not physically constrained) and has some approximation error. 
3. We feed the output of the CNN back into the physics solver and let the physics solver drop the residual and satisfy the convergence constraints.

### SURFNet: super-resolution flow network.
The core idea of SURFNet is the same as CFDNet. However, one of the drawbacks of CFDNet is that it keeps the grid size constant between data collection, training, and testing. For example, we collect data at a grid size of 64x256, we train with images of size 64x256, and predict and test for 64x256 domains. If we wished to accelerate simulations at finer grids, we should repeat this process with, for instance, 1024x1024 domains. Nevertheless, data collection and training with such large domain/image sizes is computationally impractical. 

SURFNet overcomes this burden. The idea is to accelerate fine-grid simulations while keeping coarse-grid data collection and training with coarse grid solutions. The key step of SURFNet is to _transfer learn_ the CNN weights calibrated with coarse-grid data to fine-grid datasets, which are 15x smaller than the coarse-grid training datasets. The idea is illustrated here:

![surfnet](https://user-images.githubusercontent.com/58092961/110786602-2fa7f480-8221-11eb-8e25-8ecfa475ccbd.jpg)


## Citing this work
If you believe this repository can help the development of your research:

```
@inproceedings{obiols2020cfdnet,
  title={CFDNet: A deep learning-based accelerator for fluid simulations}, 
  author={Obiols-Sales, Octavi and Vishnu, Abhinav and Malaya, Nicholas and Chandramowliswharan, Aparna},
  booktitle={Proceedings of the 34th ACM International Conference on Supercomputing},
  pages={1--12},
  year={2020}
}
```

## Install the repository and the required software

### 1. Pre-prerequisits

This repository relies on:
a) The user has `git`, `python`, and `pip` commands readily available. 
b) The Tensorflow API. The training experiments were conducted with Tensorflow 2.4 backend. Any flavour for installing the Tensorflow API is welcome (pip, conda, docker, ...). This might cause different training/inference performance values. 
c) 

### 2. Download this repository
First, download this repository:

```
git clone https://github.com/oobiols/staidy.git
```
### 2. Install required python libraries
This repository works with Python and calls several libraries that need to be installed:

```
pip install -r requirements.txt
```

## Reproduce SURFNet's results

### 1. Train the coarse model
Download the _training dataset_ and the _validation dataset_ from the following publicly available Google Drive link:

```
gdown https://drive.google.com/uc?id=1ig8gHcO6S7nM6_sC3w1tLsUh0_IhmUcI
```

After the download, you can start the training. Please note that the code will use all the available GPUs by default:

```
python coarse_model.py -bs 64 -lrt 5e-4 -e 1000 
```
The training is conducted with the EarlyStopping callback from Keras. 
