This code was tested using keras, tensorflow, and CUDA. Graphic card used to train networks was NVidia Geforce GTX 1060 6gb.
Running this project requires the IEMOCAP dataset.

Use the data_sort.py script to sort and label the data from the IEMOCAP dataset.

Then use preprocess_data.py to transform audio files into matrices.

Once the data is prepared you can run any of the train_* scripts to train the appropriate model and generate its weights.
