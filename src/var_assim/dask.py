"""Dask setup function for running on HPCs.

Adam Michael Bauer
UChicago
Feb 2026
"""

import os

from dask.distributed import Client, LocalCluster, as_completed
from multiprocessing import cpu_count


def start_dask(logger):
    """Start dask config.

    Parameters
    ----------
    logger: logging object

    Returns
    -------
    client: LocalCluster dask object
    """

    # detect if we're on slurm
    on_slurm = "SLURM_JOB_ID" in os.environ

    # if we're on slurm, set the number of CPUs based on SLURM config
    if on_slurm:
        cpus_available = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
        logger.info(
            f"    > Slurm environment detected. Using {cpus_available} allocated CPUs."
        )

    # otherwise, we're on local, so use the number of cores minus one
    else:
        cpus_available = max(1, cpu_count() - 1)  # leave 1 core free locally
        logger.info(
            f"    > Local environment detected. Using {cpus_available} of {cpu_count()} CPUs."
        )

    # initialize local dask cluseter
    cluster = LocalCluster(
        n_workers=cpus_available,
        threads_per_worker=1,
        memory_limit="auto",
        processes=True,
    )

    # setup Client
    client = Client(cluster)

    # print config details
    logger.info(f"    > Dask Client started with {cpus_available} workers.")
    logger.info(f"    > Dashboard link: {client.dashboard_link}")

    return client


def run_ensemble(c, ensemble_members, e_scat, args, runner_4dvar):
    """
    Submit ensemble members to Dask workers and collect optimized results.

    For large ensembles (n_ens > 500), submissions are batched to avoid
    overwhelming the Dask scheduler with too many in-flight futures simultaneously,
    which can cause OOM kills. Futures are released immediately after result
    collection to free scheduler memory. For smaller ensembles, all members
    are submitted at once.

    Parameters
    ----------
    c : dask.distributed.Client
        Active Dask client connected to the cluster.
    ensemble_members : list[EnsembleMember]
        List of ensemble member objects, each carrying a prior draw and
        assimilation window metadata.
    e_scat : dask.distributed.Future
        Scattered future of the EmissionsBaseline object, broadcast to all
        Dask workers.
    args : argparse.Namespace
        Parsed command-line arguments. Uses args.n_ens to determine whether
        batched submission is needed.
    runner_4dvar : callables
        The 4D-Var optimization function submitted to each Dask worker.
        Signature: runner_4dvar(member: EnsembleMember, e: EmissionsBaseline).

    Returns
    -------
    opt_ensmems : list
        List of optimized ensemble member results, in completion order
        (not submission order).
    """

    if args.n_ens > 500:
        batch_size = 200
        opt_ensmems = []
        for i in range(0, len(ensemble_members), batch_size):
            batch = ensemble_members[i : i + batch_size]
            futures = [c.submit(runner_4dvar, m, e_scat) for m in batch]
            for future in as_completed(futures):
                opt_ensmems.append(future.result())
                future.release()
    else:
        futures = [c.submit(runner_4dvar, m, e_scat) for m in ensemble_members]
        opt_ensmems = [future.result() for future in as_completed(futures)]

    return opt_ensmems


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    c = start_dask(logger)

    # check memory limits
    for w, info in c.scheduler_info()["workers"].items():
        print(w, info["memory_limit"] / 1024**3, "GB")
