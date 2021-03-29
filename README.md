# CFDNet and SURFNet
By: Octavi Obiols-Sales, Abhinav Vishnu, Nicholas Malaya, and [Aparna Chandramowlishwaran](https://hpcforge.eng.uci.edu/).

CFDNet is a deep learning-based accelerator for 2D, coarse-grid, steady-state simulations. SURFNet is an extension of CFDNet, for super-resolution of turbulent flows with transfer learning. SURFNet accelerates fine-grid problems while keeping most data collection at coarse grids.

### Why you might be interested in this repository
In this README file, you will find, first, an overview on CFDNet and SURFNet. And below, you will find different commands that will allow you to:
1. Generate a .h5 training and validation dataset file from OpenFOAM simulation data. An outline on the shape of the images in the dataset and the overall dataset structure can be found in the [paper](https://dl.acm.org/doi/pdf/10.1145/3392717.3392772?casa_token=2Vx83VWZAWwAAAAA:BauwuqoOjxXcjrpfsI1MwemUxyTb3rIfdLnf1zkUX66YCtUmdUNYWJjqf0TPYAIPDhDRX0YhwQ_0), but is also detailed [here](./datasets/datasets.md). 
2. Train the CNN with the dataset that we just generated, and create a _coarse model_.
3. Transfer learn the _coarse model_ we to high-resolution inputs.
4. Use the already trained models and reproduce the results in the paper. For such, you will OpenFOAM v8 installed. See more [here](./openfoam/).

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

# How to use

## 1. Download this repository and install all the requirements:

First, download this repository:

```
git clone https://github.com/oobiols/staidy.git
```

This repository works with Python and calls several libraries that need to be installed:

```
pip install -r requirements.txt
```

## 2. Create a training and validation dataset from OpenFOAM simulation data

The sample data for the dataset generation is in directories `train_data`, `validation_data`, `test_data`.
In each directory, we find sub-directories such as `case_1`, `case_2`, `case_3`, etc. Each `case_x` has intermediate iterations and the final steady-state solution of *one* flow configuration. Different cases have different flow configurations (same geometry at different angles of attack, a different geometry, etc.)

Run
```
python create_dataset.py --type train --turb 1 --name coarse_grid --height 32 --width 128 --grid ellipse
```

To generate a "name".h5 file, saved in ./h5_datasets/, which is a dataset that contains input X and output Y, in this case for training, where each image is of size [h,w,4]

