from models.DeepYeast_model import DeepYeastModel
from models.DeepLoc_model import DeepLocModel
from models.CP4_model import CP4Model
from models.CBDP4_model import CBDP4Model
from models.resNet_model import ResNetModel
from models.densenet_model import DenseNetModel

all_models = {
    "DeepYeast": DeepYeastModel,
    "DeepLoc": DeepLocModel,
    "CP4": CP4Model,
    "CBDP4": CBDP4Model,
    "ResNet": ResNetModel,
    "DenseNet": DenseNetModel,
}
