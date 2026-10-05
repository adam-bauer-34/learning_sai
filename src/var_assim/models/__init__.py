from .pco2geowc.runner import run_var_assim_experiment as pco2geowc_runner
from .pco2geowc_reg.runner import run_var_assim_experiment as pco2geowc_reg_runner
from .pco2geowc_reg_noic.runner import run_var_assim_experiment as pco2geowc_reg_noic_runner
from .pco2geowc_nn.runner import run_var_assim_experiment as pco2geowc_nn_runner
from .pco2geosc_reg_noic.runner import run_var_assim_experiment as pco2geosc_reg_noic_runner

from .pco2geowc3.runner import run_var_assim_experiment as pco2geowc3_runner
from .pco2geowc3_reg.runner import run_var_assim_experiment as pco2geowc3_reg_runner
from .pco2geowc3_reg_noic.runner import run_var_assim_experiment as pco2geowc3_reg_noic_runner
from .pco2geowc3_nn.runner import run_var_assim_experiment as pco2geowc3_nn_runner

MODEL_REGISTRY = {
    "pco2geowc": {"runner": pco2geowc_runner, "N_regions": "two_region"},
    "pco2geowc_reg": {"runner": pco2geowc_reg_runner, "N_regions": "two_region"},
    # q_offset: number of leading timesteps with no model error, so each
    # model-error block has N_times - q_offset entries (0 if absent)
    "pco2geowc_reg_noic": {
        "runner": pco2geowc_reg_noic_runner,
        "N_regions": "two_region",
        "q_offset": 1,
    },
    # strong-constraint counterpart of pco2geowc_reg_noic: internal variability
    # goes into the observation-error covariance instead of the control vector.
    # q_offset keeps the warm start and the prefix slicing of the shared draws
    # identical to pco2geowc_reg_noic
    "pco2geosc_reg_noic": {
        "runner": pco2geosc_reg_noic_runner,
        "N_regions": "two_region",
        "q_offset": 1,
    },
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
