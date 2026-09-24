from .pco2geowc.runner import run_var_assim_experiment as pco2geowc_runner
from .pco2geowc_reg.runner import run_var_assim_experiment as pco2geowc_reg_runner
from .pco2geowc_nn.runner import run_var_assim_experiment as pco2geowc_nn_runner

from .pco2geowc3.runner import run_var_assim_experiment as pco2geowc3_runner
from .pco2geowc3_reg.runner import run_var_assim_experiment as pco2geowc3_reg_runner
from .pco2geowc3_reg_noic.runner import run_var_assim_experiment as pco2geowc3_reg_noic_runner
from .pco2geowc3_nn.runner import run_var_assim_experiment as pco2geowc3_nn_runner

MODEL_REGISTRY = {
    "pco2geowc": {"runner": pco2geowc_runner, "N_regions": "two_region"},
    "pco2geowc_reg": {"runner": pco2geowc_reg_runner, "N_regions": "two_region"},
    "pco2geowc3": {"runner": pco2geowc3_runner, "N_regions": "three_region"},
    "pco2geowc3_reg": {"runner": pco2geowc3_reg_runner, "N_regions": "three_region"},
    "pco2geowc3_reg_noic": {
        "runner": pco2geowc3_reg_noic_runner,
        "N_regions": "three_region",
        "q_offset": 1,
    },
    "pco2geowc_nn": {"runner": pco2geowc_nn_runner, "N_regions": "two_region"},
    "pco2geowc3_nn": {"runner": pco2geowc3_nn_runner, "N_regions": "three_region"},
}
