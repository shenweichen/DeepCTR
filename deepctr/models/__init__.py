from .afm import AFM
from .autoint import AutoInt
from .ccpm import CCPM
from .dcn import DCN
from .deepfm import DeepFM
from .dien import DIEN
from .din import DIN
from .fnn import FNN
from .mlr import MLR
from .onn import ONN
from .onn import ONN as NFFM
from .nfm import NFM
from .pnn import PNN
from .wdl import WDL
from .xdeepfm import xDeepFM
from .fgcnn import FGCNN
from .dsin import DSIN
from .fibinet import FiBiNET

__all__ = ["AFM", "CCPM","DCN", "MLR",  "DeepFM",
           "MLR", "NFM", "DIN", "DIEN", "FNN", "PNN", "WDL", "xDeepFM", "AutoInt", "ONN", "FGCNN", "DSIN", "FiBiNET"]
