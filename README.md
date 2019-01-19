# Investigating deep learning algorithms in the context of Human Protein Image Classification
## Deep Learning - Fall 2018 - ETH Zurich
### Mélanie Bernhardt - Mélanie Gaillochet - Andreas Georgiou - Dimitrios Koutsoukos

## Main goal
Based on the "Human Protein Atlas Image Classication" [Kaggle competition](https://www.kaggle.com/c/human-protein-atlas-image-classification). Revolving around medical image analysis, the main goal of this project is to classify mixed patterns of proteins in microscope images using computer vision.


### Setup
- Download the data from the kaggle competition and put the training data in a folder called train and the test data in a folder called test. Both of these folders and the train.csv and sample_submission.csv files should be inside the data folder.
- Set your PYTHONPATH variable to where your code folder is:
  (i.e. `export PYTHONPATH="${PYTHONPATH}:${HOME}/DL-AtlasKaggle/code/"`)
- Set path EXP_PATH to where run results should be saved:
  (i.e. `export EXP_PATH="${HOME}/DL-AtlasKaggle/result/"`)
- Set your DATA_PATH variable to where the data is
  (ie: `export DATA_PATH="${HOME}/DL-AtlasKaggle/data/"`)

If using the Leonhard cluster (to avoid data size problem), use: `export DATA_PATH="${SCRATCH}/data/"`

- Install the requirements:
`pip install -r requirements.txt`

## Models (in 'code/models' folder)

### Data Augmentation
Data augmentation is done by rotating and reversing images. The number of transformations depends on the frequency of the image's label(s).
From 0 (most represented label class) to 8 (least represented class) transformations are possible for each image.

On 28 cores data augmentation takes about ~30 minutes. However, it has to be done separately, prior to running the models, if someone wants to later use an augmented dataset.

To augment the data, run:
`python code/data_loader/data_aug.py`
(Optional) arguments are:

- `--parallelize` to parallelize the process (if running on the cluster, for instance) (default is no parallelization)


### Baseline
- **Random Forest** <br/>
To launch training of the random forest classifier, use the following command:
`python code/mains/baseline.py -c path/to/config/<baseline json file to be used>`.
This will automatically launch prediction

### Simple CNN with increasing complexity
- **CP4_model**: Simple *4-layer* CNN (with ReLU activation and max pooling)
- **CBDP4_model**: 4-layer CNN (each layer is followed by a *batch_normalization* layer, and then by a dropout layer with rate 0.5 before the max pooling layer)

### Existing state-of-the-art models
- **DeepYeast**: 11-layer CNN, with 8 convolutional layers and 3 fully connected layers
- **DeepLoc**: 11-layer CNN, with 8 convolutional layers and 3 fully connected layers
- **Resnet**: Residual Convolutional Networks {18, 34, 50, 101, 152, 200} variants
- **Densenet**: Densely Connected Convolutional Networks {121, 169, 201} variants

Parameters for each model can be modified in the corresponding file of the 'code/configs' folder.

## Training the models
Each training procedure is defined by one JSON configuration file.
In this file you specify the following arguments:

- "model": (mandatory) to choose between "DeepYeast, "CP4, "CBDP4", "ResNet", "DenseNet"
- "learning_rate": (mandatory) learning rate for the ADAM optimizer
- "max\_to\_keep": (mandatory) max number of model checkpoints to keep
- "exp_name": (mandatory) name of the folder in which the checkpoint and summary subfolder for this training run are going to be placed
- "val_split": (optional, default: 0.1) validation / train split ratio to use
- "num_epochs": number of epochs to train your model
- "batch_size": batch size to use
- "use\_weighted\_loss": (optional, default: false) whether to use class weigths to weight the loss function
- "input\_size": (optional, default: 512) if you want to resize the input images to "input\_size" in each dimension
- "f1_loss": (optional, default: false) whether to use the f1 loss instead of the cross-entropy loss
- "focal_loss": (optional, default: false) whether to use the focal loss instead of the cross-entropy loss
- "augment": (optional, default: false) whether to use the augmented dataset
- "resnet_size": (optional, default 101) the depth of the Residual Network in case you are using one (you can choose from the {18, 34, 50, 101, 152, 200} variants
- "densenet_size": (optional, default 121) the depth of the Dense Network in case you are using one (you can choose from the {121, 169, 201} variants

Then to launch training use the following command:
`python code/mains/train_main.py -c path/to/config/<json file to be used>`

To reproduce the experiments of the report you can use the config files in the `code/configs/final_exp` subfolder.

## Predicting from a trained model
To output a csv prediction file for the images in the Kaggle test set use the `predict_main` file. You also have to feed the training config file as a parser argument. If you are running the prediction code on the same machine that was used for training you don't need to specify the number of the model checkpoint to use, it will automatically retrieve the latest checkpoint saved during training. However if you are on an other machine (i.e. training on the cluster, downloading the checkpoint folder and predicting on your laptop) you have to use an additional parser argument `-check_nb` that specifies the number of the checkpoint to use for prediction. 

**Note**: the checkpoints are saved as `-{check_nb}.meta` files in the checkpoint subfolder of the training experiment folder.

Example of command to launch prediction:
`python code/mains/predict_main.py -c "path/to/config/<json file to be used>" -check_nb 11900`

## Averaging probabilities from several models
If you have several trained models and you wish to combine all predicted probabilities (by averaging them) in order to predict the labels you can use the `predict_from_several_main.py` file. It takes a list of config files (one per model to load) and a corresponding list of check_nb checkpoints to load.
The result are saved in a csv file called `/{filename}.csv` in your `EXP_PATH` folder. You can also specify `filename` via the `-om` parser argument.

Example of such a command:
`python code/mains/predict_from_several_main.py -c "path/to/config1 path/to/config2" -check_nb "checknb1 checknb2" -om "filename"`

## Acknowledgements
Tensorflow template taken from [here](https://github.com/jtoy/awesome-tensorflow).

Please see the separate README in the code folder for instructions on how to use it.


