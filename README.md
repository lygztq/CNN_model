# CNN_model
this is a model for CNN.
## About Directorys
* dataSet: This directory contains bin file for train set and test set.
* models: This directory contains the saved model for CNN.
* ResultPNG: This directory contains the source dataset in png format.
* totalDataSet: This directory contains the total dataset without labels, which for prectical use.
* ResultPNGGray: This directory is used for test.

## How To Use
Put your train data and test data into the ResultPNG directory, run data_process.py, it will generate bin files for your data. Then run cnn_train.py to train the CNN model, the parameters and graph will be saved in models. You can change the hyper parameters in cnn.py.
