# DL-AtlasKaggle

Based on the "Human Protein Atlas Image Classication" [Kaggle competition] (https://www.kaggle.com/c/human-protein-atlas-image-classification)
Revolving around medical image analysis, the main goal of this project is to classify mixed patterns of proteins in microscope images using computer vision.

## Setup
- Set your PYTHONPATH variable to where your code folder is (ie: `export DATA_PATH="${HOME}/DL-AtlasKaggle/data/"`)
- Set path EXP_PATH to where run results should be saved (ie: `export PYTHONPATH="${PYTHONPATH}:${HOME}/DL-AtlasKaggle/code/"`)
- Set your DATA_PATH variable to where the data is (ie: `export EXP_PATH="${HOME}/DL-AtlasKaggle/result/"`)
If using cluster (to avoid data size problem), use: export `DATA_PATH="${SCRATCH}/"`

- Install the requirements:
`pip install -r requirements.txt`

## Models (in 'code/models' folder)
### Simple CNN with increasing complexity
- *CP2_model* : Simple *2-layer* CNN (with ReLU activation and max pooling)
- *CP4_model*: *4-layer* CNN
- *CDP4_model*: 4-layer CNN (each followed by *dropout* of rate 0.5 before max pooling)
- *CBDP4_model*: 4-layer CNN (each with *batch_normalization*, followed by dropout of rate 0.5 before max pooling)
- *SimpleCNN_model*: 3-layer CNN (each with *batch_normalization*, followed by dropout of rate 0.5 before max pooling)

### Existing state-of-the-art models
- *DeepYeast_model*: DeepYeast (11-layer CNN, with 8 convolutional layers and 3 fully connected layers)
- *inception_model*: InceptionNets
- *resNet_model*: ResNets
- DenseNets

Parameters for each model can be modified in the corresponding file of the 'code/configs' folder.

## Running the Models
`python code/mains/train_main.py -c code/configs/<json file to be used>`









