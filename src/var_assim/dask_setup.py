"""Dask setup function for running on HPCs.

Adam Michael Bauer
UChicago
Feb 2026
"""

import os 

from dask.distributed import Client, LocalCluster

def start_dask(logger):
    """Start dask config.

    Parameters
    ----------
    logger: logging object

    Returns
    -------
    client: LocalCluster dask object
    """

    # Slurm sets the SLURM_CPUS_PER_TASK variable
    # We use this to tell Dask exactly how many workers to start
    cpus_available = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    
    # Initialize a cluster that matches the Slurm allocation
    cluster = LocalCluster(
        n_workers=cpus_available, 
        threads_per_worker=1,  # 1 thread per worker is best for heavy math
        memory_limit='auto',
        processes=True
    )
    client = Client(cluster)
    
    # add logs
    logger.info(f"Dask Client started with {cpus_available} workers.")
    logger.info(f"Dashboard link: {client.dashboard_link}")

    return client