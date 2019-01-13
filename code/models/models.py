from models.DeepYeast_model import DeepYeastModel
from models.CP2_model import CP2Model
from models.CP4_model import CP4Model
from models.CDP4_model import CDP4Model
from models.CBDP4_model import CBDP4Model
from models.CDP2_model import CDP2Model
from models.CBDP2_model import CBDP2Model
from models.SimpleCNN_model import SimpleCNNModel
from models.inception_model import InceptionModel
from models.resNet_model import ResNetModel
from models.densenet_model import DenseNetModel
from models.kaggle_model import KaggleModel
from models.DeepSimple_model import DeepSimpleModel

all_models = {
    "DeepYeast": DeepYeastModel,
    "SimpleCNN": SimpleCNNModel,
    "CP2": CP2Model,
    "CP4": CP4Model,
    "CDP4": CDP4Model,
    "CBDP4": CBDP4Model,
    "CDP2": CDP2Model,
    "CBDP2": CBDP2Model,
    "Inception": InceptionModel,
    "ResNet": ResNetModel,
    "DenseNet": DenseNetModel,
    "Kaggle": KaggleModel,
    "DeepSimple": DeepSimpleModel
}
