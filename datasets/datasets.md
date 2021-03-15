# Dataset creation

The basic unit of the datasets for CFDNet and SURFNet are its input-output pairs. The input to the CNN and the output (target) of the CNN are images of equal size. Each image is constructed as follows:


This image has size MxNxZ, where:
1. MxN is the domain size, that is, the 2D computational cells of the physical domain, where the value of the corresponding variables are computed. 
2. Z, the number of primary variables of the problem. In our test cases, Z=4 - velocity in the first cartesian direction X, velocity in the second cartesian direction Y, pressure, and the eddy viscosity. However, this image representation is amenable to other flow configurations that take into account other primary variables, such the density in compressible flows, or the temperature in heat transfer problems. 

Once the training dataset is created, we use it to train the CNN as follows:




As stated, the input and the output are images of equal size. The only difference between the input and the output is the value of the primary variables that they contain. In the input, the values are those of any intermediate iteration, whereas the output has the values of the steady-state final solution. 
