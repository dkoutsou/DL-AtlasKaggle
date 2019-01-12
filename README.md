# DL-AtlasKaggle
## Deep Learning - Fall 2018 - ETH Zurich
### Mélanie Bernhardt - Mélanie Gaillochet - Andreas Georgiou - Dimitrios Koutsoukos

## REMARK: we have a second readme for the template, should we delete it? But we should probalby cite the template anyway.

## Main goal
Based on the "Human Protein Atlas Image Classication" Kaggle competition (https://www.kaggle.com/c/human-protein-atlas-image-classification) <br/>
Revolving around medical image analysis, the main goal of this project is to classify mixed patterns of proteins in microscope images using computer vision.


### Setup
- Set your PYTHONPATH variable to where your code folder is <br/>
(ie: `export DATA_PATH="${HOME}/DL-AtlasKaggle/data/"`)
- Set path EXP_PATH to where run results should be saved <br/>
(ie: `export PYTHONPATH="${PYTHONPATH}:${HOME}/DL-AtlasKaggle/code/"`)
- Set your DATA_PATH variable to where the data is <br/>
(ie: `export EXP_PATH="${HOME}/DL-AtlasKaggle/result/"`)<br/>
If using cluster (to avoid data size problem), use: export `DATA_PATH="${SCRATCH}/"`

- Install the requirements:
`pip install -r requirements.txt`

## Models (in 'code/models' folder)
*BEFORE SUBMISSION DELETED THE UNUSUED MODELS*

### Baseline
- **Random Forest**

### Simple CNN with increasing complexity
- **CP2_model** : Simple *2-layer* CNN (with ReLU activation and max pooling)
- **CP4_model**: Simple *4-layer* CNN(with ReLU activation and max pooling)
- **CDP4_model**: 4-layer CNN (each followed by *dropout* of rate 0.5 before max pooling)
- **CBDP4_model**: 4-layer CNN (each with *batch_normalization*, followed by dropout of rate 0.5 before max pooling)
- **SimpleCNN_model**: 3-layer CNN (each with *batch_normalization*, followed by dropout of rate 0.5 before max pooling) *NOT USED FOR NOW*
_ **DeepSimple_model**: 3 blocks of (conv layer + batch_normalization + dropout + conv layer + batch_normalization + dropout+ 2x2 maxpooling) followed by one dense layer with 28 output units.

### Existing state-of-the-art models
- **DeepYeast_model**: DeepYeast (11-layer CNN, with 8 convolutional layers and 3 fully connected layers)
- **inception_model**: InceptionNets
- **resNet_model**: ResNets
- DenseNets

Parameters for each model can be modified in the corresponding file of the 'code/configs' folder.

## Training the models
Each training procedure is defined by one jSON configuration file.
In this file you specify the following arguments:
- "model": (mandatory) to choose between "DeepYeast, "SimpleCNN" (*maybe to remove*), "CP2", "CP4, "CDP4", "CBDP4", "CDP2", "CBDP2", "Inception", "ResNet", "Kaggle", "DeepSimple".
- "learning_rate": (mandatory) learning rate for ADAM
- "max_to_keep": (mandatory) max number of model checkpoint to keep
- "exp_name": (mandatory) name of the folder in which to place the checkpoint and summary subfolder for this training run.
- "val_split": (optional, default: 0.1) validation / train split ratio to use
- "num_epochs": number of epochs to train your model
- "batch_size": batch size to use.
- "use_weighted_loss": (optional, default: false) whether to use class weigths to weight to loss function.
- "input_size": (optional, default: 512) if you want to resize the input images to "input_size"x"input_size".
- "use_f1_loss": (optional, default: false) whether to use f1 loss instead of cross-entropy loss.

Then to launch training use the following command:
`python code/mains/train_main.py -c path/to/config/<json file to be used>`

If you want to reproduce the experiments of the report you can use the config files in the `code/configs/final_exp` subfolder.

## Predicting from a trained model
To output a csv prediction file for the images in the Kaggle test set use `predict_main`. You also have to feed the training config file as a parser argument. If you are running the prediction code on the same machine that was used for training you don't need to specify the number of the model checkpoint to use, it will automatically retrieve the latest checkpoint saved during training. However if you are on a other machine (i.e. training on the cluster, downloading the checkpoint folder and predicting on local laptop) you have to use an additional parser argument `-check_nb` that specifies the number of the checkpoint to use for prediction. Note: the checkpoints are saved as `-{check_nb}.meta` files in the checkpoint subfolder of the training experiment folder.

Example of command to launch prediction
`python code/mains/predict_main.py -c "path/to/config/<json file to be used>" -check_nb 11900`

## Averaging probabilities from several models
If you have several trained models and you wish to combine all predicted probabilities (averaging them) in order to predict the labels you can use `predict_from_several_main.py`. It takes a list of config files (one per model to load) and a corresponding list of check_nb for the corresponding checkpoint numbers to load.
The result are saved in a csv file called `/mean_pred.csv` in your `EXP_PATH` folder.
Example of command:
`python code/mains/predict_from_several_main.py -c "path/to/config1 path/to/config2" -check_nb "checknb1 checknb2"`


