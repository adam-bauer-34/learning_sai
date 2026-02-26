"""Dask setup function for running on HPCs.

Adam Michael Bauer
UChicago
Feb 2026
"""

import os 

from dask.distributed import Client, LocalCluster
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
        logger.info(f"Slurm environment detected. Using {cpus_available} allocated CPUs.")

    # otherwise, we're on local, so use the number of cores minus one
    else:
        cpus_available = max(1, cpu_count() - 1)  # leave 1 core free locally
        logger.info(f"Local environment detected. Using {cpus_available} of {cpu_count()} CPUs.")

    # initialize local dask cluseter
    cluster = LocalCluster(
        n_workers=cpus_available,
        threads_per_worker=1,
        memory_limit='auto',
        processes=True
    )

    # setup Client
    client = Client(cluster)

    # print config details
    logger.info(f"Dask Client started with {cpus_available} workers.")
    logger.info(f"Dashboard link: {client.dashboard_link}")

    return client