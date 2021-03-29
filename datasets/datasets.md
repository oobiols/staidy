## Dataset overview

The basic unit of the datasets for CFDNet and SURFNet are its input-output pairs. The input to the CNN and the output (target) of the CNN are images of equal size. Each image is constructed as follows:

![inputoutput](https://user-images.githubusercontent.com/58092961/111132371-81f55800-8536-11eb-8c87-61402d8d8813.jpg)


This image has size MxNxZ, where:
1. MxN is the domain size, that is, the 2D computational cells of the physical domain, where the value of the corresponding variables are computed. 
2. Z, the number of primary variables of the problem. In our test cases, Z=4 - velocity in the first cartesian direction X, velocity in the second cartesian direction Y, pressure, and the eddy viscosity. However, this image representation is amenable to other flow configurations that take into account other primary variables, such the density in compressible flows, or the temperature in heat transfer problems. 

Once the training dataset is created, we use it to train the CNN as follows:

![network](https://user-images.githubusercontent.com/58092961/111132666-dc8eb400-8536-11eb-8b0f-be0c31e98bb7.jpg)

As stated, the input and the output are images of equal size. The only difference between the input and the output is the value of the primary variables that they contain. In the input, the values are those of any intermediate iteration, whereas the output has the values of the steady-state final solution. 

## Dataset generation

We provide the recipe and a tutorial with data to generate an example of training, validation, and test dataset.

The recipe (create_dataset.py) creates the input (X) and the output (Y) of one dataset.

X: of size (S,M,N,Z), where S is the number of samples, that is, the number of intermediate iterations from different flow configurations that use the same MxN domain size and the same primary variables Z.

Y: of size (S,M,N,Z), where S is the number of samples, but in this case, the steady-state solution to which each intermediate iteration maps to.

## Tutorial




