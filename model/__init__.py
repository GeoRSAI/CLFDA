from importlib import import_module
from .SEANet import seanet
from .load_r3mamba import unetformer, rs3mamba
from .load_r3mamba import rsm_ss, CLFDA



model_dict = {
    "unetformer":unetformer,
    "seanet": seanet,
    "rs3mamba": rs3mamba,
    "rsm_ss": rsm_ss,
    "CLFDA":CLFDA,
}