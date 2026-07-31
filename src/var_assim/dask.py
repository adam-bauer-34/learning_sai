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

    Results are returned in SUBMISSION order, so opt_ensmems[i] is always the
    optimized form of ensemble_members[i].

    Collection still happens via as_completed, so each result is gathered and its
    future released as soon as that member finishes -- the reason as_completed is
    used here in the first place. Only the slot each result is written into
    changes. Appending in completion order instead makes ens_mem in the output a
    label for "whichever member finished i-th", which varies between assimilation
    windows, between models, and between runs of the same config. Nothing is lost
    or duplicated by that -- each member's record stays internally coherent -- but
    a member cannot be followed across windows, and member-level results are not
    reproducible.

    Returns
    -------
    opt_ensmems : list
        List of optimized ensemble member results, in submission order, i.e.
        aligned with `ensemble_members`.
    """

    def _collect_in_order(futures):
        """Gather futures as they complete, into submission-order slots."""

        # future.key -> submission index. keys are unique because submit is called
        # with pure=False; see the note below on why that matters.
        slot_of = {future.key: i for i, future in enumerate(futures)}

        results = [None] * len(futures)
        for future in as_completed(futures):
            results[slot_of[future.key]] = future.result()
            future.release()

        return results

    # NOTE: pure=False. Client.submit defaults to pure=True, which hashes the
    # arguments and hands back the *same* future for an identical submission. Two
    # ensemble members carrying identical priors would then collapse into a single
    # task, silently shrinking the ensemble. Random draws make that vanishingly
    # unlikely, but pure=False also guarantees the distinct keys that the
    # submission-order mapping above relies on, so both properties hold by
    # construction rather than by luck.
    if args.n_ens > 500:
        batch_size = 200
        opt_ensmems = []
        for i in range(0, len(ensemble_members), batch_size):
            batch = ensemble_members[i : i + batch_size]
            futures = [c.submit(runner_4dvar, m, e_scat, pure=False) for m in batch]
            opt_ensmems.extend(_collect_in_order(futures))
    else:
        futures = [
            c.submit(runner_4dvar, m, e_scat, pure=False) for m in ensemble_members
        ]
        opt_ensmems = _collect_in_order(futures)

    return opt_ensmems


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    c = start_dask(logger)

    # check memory limits
    for w, info in c.scheduler_info()["workers"].items():
        print(w, info["memory_limit"] / 1024**3, "GB")
