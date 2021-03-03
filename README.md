# CFDNet and SURFNet
CFDNet is a deep learning-based accelerator for 2D, coarse-grid, steady-state simulations. SURFNet is an extension of CFDNet, which targets fine-grid problems with transfer learning. SURFNet accelerates fine-grid problems while keeping most data collection at coarse grids. 

# CFDNet: accelerating steady-state simulations
CDFNet's idea and architecture can be found in our 2020 publication: 
https://dl.acm.org/doi/pdf/10.1145/3392717.3392772?casa_token=EpdKLxlSgXkAAAAA:Ozdj7mt64YjVg3cbmqhS3t6otH9Zr5krINL8rqT7PX4THitij6fDV7IdGOdpPDSAFSv-hmk7yL3U

CFDNet is a framework that accelerates steady-state simulations. We summarize the idea as follows: 
1. We start the steady-state simulation as usual, with a user-given initial condition, and let the solver carry K iterations -- K _warmup_ iterations, as we call it. Once we reach this intermidiate state, we create our input representation using the flow variable values at that K iteration.
2. We create the input representation described in the publication and pass it to the CNN. The CNN then infers the steady-state -- which is unconstrained (not physically constrained) and has some approximation error. 
3. We feed the output of the CNN back into the physics solver and let the physics solver drop the residual and satisfy the convergence constraints.


# About this repo (A)
This repo is mostly about bullet number 2 of the former list. Hence, the code found here has, at first, the following objectives:
1. Create the input-output representation to train the CNN.
2. Train the CNN 
3. Predict the steady-state and evaluate its quality quantitatively.

# SURFNet: super-resolution flow network.
The core idea of SURFNet is the same as CFDNet, so we will not list the steps again. However, one of the drawbacks of CFDNet is that it keeps the grid size constant between data collection, training, and testing. For example, we collect data at a grid size of 64x256, we train with images of size 64x256, and predict and test for 64x256 domains. If we wished to accelerate simulations at finer grids, we should repeat this process with, for instance, 1024x1024 domains. Nevertheless, data collection and training with such large domain/image sizes is computationally impractical. 

SURFNet overcomes this burden. The idea is to accelerate fine-grid simulations while keeping coarse-grid data collection and training with coarse grid solutions. The key step of SURFNet is to _transfer learn_ the CNN weights calibrated with coarse-grid data to fine-grid datasets, which are 15x smaller than the coarse-grid training datasets.

# About this repo (B)
Together with the functions to create the dataset, train the CNN with coarse-grid data, and predict steady-state fields, the code has an additional objective:
1.  Transfer the weights trained with coarse-grid data and re-train the CNN for very few epochs using tiny, fine-grid datasets.
