This readme is about the train.py file and the train-generator.py file

The difference between both files is in the way the training dataset is loaded to main memory.

- In train.py, the entire training dataset is loaded to main memory, in variables X_train and Y_train for the input and the output, respectively.

- In train-generator.py, we use a generator to load one batch at a time to main memory. This is useful for when the training dataset does not fit in main memory.


Both files consist of the following steps:

1. Set the path to and load the coarse-grid training dataset.
2. Set the path to and load the coarse-grid validation dataset.
3. Setup the CNN that will be trained and will become the coarse model. The CNN is setup using our own class NeuralNetwork, which can be found in func/models.py
4. Train the CNN in a distributed manner (using all available GPUs).
5. Save the model, plot the history, and write the history values into a txt file. 
