"""Grid of angle bias / percent error over assimilation windows.

Script version of `analysis/notebooks/grid-bias-error.ipynb`. Four rows, one
per experiment configuration:
    - Row 1: noisy model, ssp245, 0.1 deg / dec
    - Row 2: noisy model, ssp585, 0.1 deg / dec
    - Row 3: noisy model, ssp245, 0.2 deg / dec
    - Row 4: no-noise model (`<model_config>_nn`), ssp245, 0.1 deg / dec

The noisy model name is built as `<model_config>[_reg][_noic]` from the
`--reg_noise` and `--no_ic` flags.

To run:
    python analysis/scripts/grid_bias_error.py --model_config pco2geowc3
        [--reg_noise] [--no_ic] [--ramp_rate linear] [--no-save] ...

Post-processing I/O only, safe to run interactively.

Adam Michael Bauer
UChicago
"""

import argparse
import gc

import matplotlib.pyplot as plt
import numpy as np
import yaml
from datatree import open_datatree
from scipy.stats import describe

from var_assim.calibration.noise import ClimateModelNoise
from var_assim.calibration.priors import ClimateModelPriors
from var_assim.calibration.truth import ClimateModelTruth
from var_assim.calibration.windowing import AssimilationWindowing
from var_assim.config import DATA_DIR_ABS, NOISE_PATH, PRIOR_PATH, TRUTH_PATH, WINDOW_PATH
from var_assim.plotting.bias_perc_error_over_time import make_grid_plot
from var_assim.plotting.filtering import filter_by_max_conceivable, filter_datatree_by_cost_ratio_memeff
from var_assim.models import MODEL_REGISTRY
from var_assim.plotting.pproc import get_angle_r2, get_angle_r3
from var_assim.plotting.presets import get_presets

MODEL_CONFIGS = ["pco2geowc", "pco2geowc3"]
N_REGIONS = {"two_region": 2, "three_region": 3}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_config", required=True, choices=MODEL_CONFIGS,
                    help="base model name")
    ap.add_argument("--reg_noise", action="store_true",
                    help="use regional-noise runs (appends _reg to the noisy model name)")
    ap.add_argument("--no_ic", action="store_true",
                    help="use no-IC runs (appends _noic to the noisy model name)")

    # defaults are the values currently in the notebook
    ap.add_argument("--prefix", default="var-assim-output", help="output file prefix")
    ap.add_argument("--noise_model", default="AR1",
                    help="internal-variability noise model for the noisy rows")
    ap.add_argument("--thetas", nargs="+", default=["5", "10", "15", "20", "25", "30"],
                    help="true angles (as in the file names)")
    ap.add_argument("--ecs", type=float, default=3.0)
    ap.add_argument("--nyrsramp", type=int, default=50)
    ap.add_argument("--windowing", default="gradual", help="windowing config")
    ap.add_argument("--nens", type=int, default=1000)
    ap.add_argument("--tmin", type=int, default=2025)
    ap.add_argument("--ramp_rate", default="linear")
    ap.add_argument("--angle_max", type=float, default=90.,
                    help="maximum conceivable angle; larger values are filtered")
    ap.add_argument("--cost_threshold", type=float, default=5e1,
                    help="cost-ratio screening threshold")
    ap.add_argument("--plo", type=int, default=5, help="lower percentile")
    ap.add_argument("--phi", type=int, default=95, help="upper percentile")
    ap.add_argument("--ylim_offset", type=float, default=5)
    ap.add_argument("--no-save", action="store_true", help="do not write the figure")
    ap.add_argument("--show", action="store_true", help="show the figure interactively")
    return ap.parse_args()


def build_panel_configs(cli):
    """Per-row experiment properties."""

    model = cli.model_config
    if cli.reg_noise:
        model += "_reg"
    if cli.no_ic:
        model += "_noic"
    nn_model = f"{cli.model_config}_nn"

    noisy = dict(prefix=cli.prefix, model=model, noise_model=cli.noise_model,
                 reg_noise=cli.reg_noise)
    panel_configs = {
        f"{model}_ssp245": dict(noisy, ssp="ssp245", degpdec=0.1),
        f"{model}_ssp585": dict(noisy, ssp="ssp585", degpdec=0.1),
        f"{model}_ssp245_repeat": dict(noisy, ssp="ssp245", degpdec=0.2),
        f"{nn_model}_ssp245": dict(prefix=cli.prefix, model=nn_model, noise_model="nn",
                                   reg_noise=False, ssp="ssp245", degpdec=0.1),
    }
    return model, panel_configs


def build_row_titles(cli, panel_configs):
    ssp_names = {"ssp245": "SSP2-4.5", "ssp585": "SSP5-8.5"}
    titles = []
    for pc in panel_configs.values():
        if pc["noise_model"] == "nn":
            iv = "No Internal Variability"
        else:
            iv = f"{pc['noise_model'].replace('AR', 'AR(')}) Internal Variability"
        reg = "AR(0) Regional Noise" if pc["reg_noise"] else "No Regional Noise"
        titles.append(
            rf"{ssp_names.get(pc['ssp'], pc['ssp'])} | $\Delta$ T / decade = "
            rf"{pc['degpdec']} $^\circ$C / decade | {iv} | {reg}"
        )
    return titles


def output_fname(cli, pc, th):
    noise_tag = f"{pc['noise_model']}+reg" if pc["reg_noise"] else pc["noise_model"]
    return DATA_DIR_ABS / "output" / pc["model"] / (
        f"{pc['prefix']}_{pc['ssp']}_{pc['model']}_{cli.windowing}_{noise_tag}"
        f"_TMIN{cli.tmin}_THETA{th}_ECS{cli.ecs}_ramprate{cli.ramp_rate}"
        f"_DEGpDEC{pc['degpdec']}_NYRSRAMP{cli.nyrsramp}_Nens{cli.nens}.nc"
    )


def n_regions(model):
    return N_REGIONS[MODEL_REGISTRY[model]["N_regions"]]


def get_angle(alphas, betas, l, e, g):
    """SAI inequality angle using every region of the model, so the 3-region
    model uses get_angle_r3 (get_angle_r2 would silently drop region 3)."""

    angle_fn = get_angle_r3 if len(alphas) == 3 else get_angle_r2
    # both return a length-1 array, so reduce to a scalar
    return np.ravel(angle_fn(*alphas, *betas, l, e, g))[0]


def angles_from_controls(ctrl, n, nreg):
    """Angle for each of the first `n` members of a `controls`-like DataArray."""

    regs = range(1, nreg + 1)
    a = [ctrl.sel(vari=f"ALPHA_R{r}").values for r in regs]
    b = [ctrl.sel(vari=f"BETA_R{r}").values for r in regs]
    l, e, g = (ctrl.sel(vari=name).values for name in ("L", "EPS", "G"))
    return np.array([
        get_angle([x[i] for x in a], [x[i] for x in b], l[i], e[i], g[i])
        for i in range(n)
    ])


def main():
    cli = parse_args()

    presets, _ = get_presets()
    plt.rcParams.update(presets)

    model_prefix, panel_configs = build_panel_configs(cli)
    row_titles = build_row_titles(cli, panel_configs)

    # get the assimilation windows for this config
    with open(WINDOW_PATH) as f:
        window_config = yaml.safe_load(f)
    windows = window_config[cli.windowing]["windows"]

    # check inputs up front
    missing = [str(output_fname(cli, pc, th))
               for pc in panel_configs.values() for th in cli.thetas
               if not output_fname(cli, pc, th).exists()]
    if missing:
        raise FileNotFoundError("missing input files:\n" + "\n".join(missing))

    angles = np.zeros((len(panel_configs), len(cli.thetas), len(windows), cli.nens))
    angles_tr = np.zeros(len(cli.thetas))
    angles_prior = np.zeros(cli.nens)
    dropped_vars = None

    for idx, pc in enumerate(panel_configs.values()):
        for angx, th in enumerate(cli.thetas):
            print(f"Processing {pc['model']} {pc['ssp']} DEGpDEC{pc['degpdec']} THETA{th}")

            # make temporary CLI namespace for processing
            cli_args = argparse.Namespace(
                model=pc["model"],
                ssp=pc["ssp"],
                noise_model=pc["noise_model"],
                tmin=cli.tmin,
                theta=float(th),
                AR=1 if pc["noise_model"] != "nn" else None,
                ecs=cli.ecs,
                DEGpDEC=pc["degpdec"],
                NYRSRAMP=cli.nyrsramp,
                Nens=cli.nens,
                reg_noise=pc["reg_noise"],
                windowing=cli.windowing,
            )

            # setup data classes
            Noise = ClimateModelNoise.from_cli_and_yaml(cli_args, NOISE_PATH)
            ClimateModelPriors.from_cli_and_yaml_and_noise(cli_args, PRIOR_PATH, Noise)
            Windowing = AssimilationWindowing.from_cli_and_yaml(cli_args, WINDOW_PATH)
            Truth = ClimateModelTruth.from_cli_and_yaml(cli_args, TRUTH_PATH)

            # compute true value of angle
            nreg = n_regions(pc["model"])
            if idx == 0:
                angles_tr[angx] = get_angle(
                    Truth.ALPHA_TR, Truth.BETA_TR,
                    Truth.L_TR, Truth.EPS_TR, Truth.G_TR,
                )

            fname = output_fname(cli, pc, th)

            # variables we need for analysis (constant across configs)
            if dropped_vars is None:
                dt_meta = open_datatree(fname, engine="netcdf4")
                relevant_vars = ["cost_hist", "controls", "controls_hist"]
                dropped_vars = [v for v in dt_meta[str(windows[0])].ds.data_vars
                                if v not in relevant_vars]
                for node in dt_meta.subtree:
                    if node.ds is not None:
                        node.ds.close()
                del dt_meta

            # open and clean dataset
            dt = open_datatree(fname, engine="netcdf4", drop_variables=dropped_vars)
            dt_clean = filter_datatree_by_cost_ratio_memeff(
                dt, threshold=cli.cost_threshold, clear_after_each_node=True)

            # make priors (only have to do once)
            if idx == 0 and angx == 0:
                prior_ctrl = dt[str(windows[0])].ds.controls_hist.sel(iter=0)
                angles_prior[:] = angles_from_controls(prior_ctrl, cli.nens, nreg)

                print("Angle prior summary statistics:")
                for key, value in describe(angles_prior)._asdict().items():
                    print(f"{key:>10}: {value}")

            # compute angles
            winds = np.array([w[1] for w in Windowing.windows])
            for w_, w in enumerate(winds):
                angles[idx, angx, w_] = angles_from_controls(
                    dt_clean[str(w)].ds.controls, cli.nens, nreg)

            # cleanup
            for node in dt.subtree:
                if node.ds is not None:
                    node.ds.close()
            del dt, dt_clean
            gc.collect()

    print(f"True values of the theta parameter are: {angles_tr}")
    angles_filtered = filter_by_max_conceivable(angles, cli.angle_max)

    make_grid_plot(
        angles, angles_tr, angles_prior, cli.thetas,
        winds, PLO=cli.plo, PHI=cli.phi,
        row_titles=row_titles,
        save_figs=not cli.no_save,
        fname=(f"grid-bias-error-{model_prefix}-{cli.windowing}-TMIN{cli.tmin}"
               f"-Nens{cli.nens}-{cli.plo}-{cli.phi}range-{cli.ramp_rate}_claude"),
        YLIM_OFFSET=cli.ylim_offset,
    )

    if cli.show:
        plt.show()


if __name__ == "__main__":
    main()
