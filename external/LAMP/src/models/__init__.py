from src.models.baseline.a3tgcn import A3TGCN
from src.models.baseline.dcrnn import DCRNN
from src.models.baseline.dyngesn import DynGESN
from src.models.baseline.evolve_gcn import EvolveGCN_O, EvolveGCN_H
from src.models.baseline.gclstm import GCLSTM
from src.models.baseline.gcrn import (
    GCRN_LSTM,
    GCRN_GRU,
)
from src.models.baseline.tgcn import TGCN
from src.models.lamp import LAMP

MODELS = {
    "LAMP": LAMP,
    "TGCN": TGCN,
    "DCRNN": DCRNN,
    "GCRN_LSTM": GCRN_LSTM,
    "GCRN_GRU": GCRN_GRU,
    "A3TGCN": A3TGCN,
    "DynGESN": DynGESN,
    "GCLSTM": GCLSTM,
    "EvolveGCN_O": EvolveGCN_O,
    "EvolveGCN_H": EvolveGCN_H,
}
