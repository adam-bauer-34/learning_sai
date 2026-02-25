from .pco2geowc import run_var_assim_experiment as pco2geowc_runner
from .pco2geowc3 import run_var_assim_experiment as pco2geowc3_runner
from. pco2geowc_nn import run_var_assim_experiment as pco2geowc_nn_runner

MODEL_REGISTRY = {
    'pco2geowc': pco2geowc_runner,
    'pco2geowc3': pco2geowc3_runner,
    'pco2geowc_nn': pco2geowc_nn_runner
}