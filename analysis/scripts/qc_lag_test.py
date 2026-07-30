"""QC: lag cross-correlation of recovered model errors against the truth.

A misindexed model-error column in the TLM does not make the assimilation fail
or even look obviously wrong -- the optimizer still converges, and the posterior
model errors still have a plausible magnitude and spectrum. What it does is
assign each recovered model error to the wrong year. That is invisible in any
summary statistic of the posterior, and it is exactly what this test looks for.

For each model-error block, correlate the recovered model errors against the
truth at a range of lags. A correct assimilation peaks at lag 0. A one-year
misassignment peaks at lag +1, which is what production output showed before the
TLM column indices were fixed (qR1 read +0.03 at lag 0 and +0.94 at lag +1).

The global block qAT is the least sensitive version of this test, because it is
AR(1) and temporally smooth: a one-step shift still leaves it correlated at lag
0, so the signature is a lower peak rather than a displaced one. The regional
blocks enter the model instantaneously and are white in time, so for those the
peak location is unambiguous.

Usage
-----
    python analysis/scripts/qc_lag_test.py <path to output .nc> [--group 2100]

Adam Michael Bauer
UChicago
"""

import argparse
import re
import sys

import numpy as np

from datatree import open_datatree

# lags to scan, in timesteps (years). +/-2 is enough to see a one-step shift and
# to confirm the peak is a peak rather than the edge of a plateau.
LAGS = (-2, -1, 0, 1, 2)

# a block whose truth is essentially constant carries no timing information, so
# correlating against it is meaningless. this catches the inert regional noise
# that the inv=True prior bug used to produce.
MIN_TRUTH_STD = 1e-6


def _lag_corr(recovered, truth, lag):
    """Correlate recovered against truth shifted by `lag` timesteps.

    A positive lag compares recovered[t] with truth[t + lag], so a peak at
    lag +1 means the recovered series is one step behind the truth.
    """

    if lag > 0:
        a, b = recovered[:-lag], truth[lag:]
    elif lag < 0:
        a, b = recovered[-lag:], truth[:lag]
    else:
        a, b = recovered, truth

    if len(a) < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def _find_blocks(varis):
    """Group the model-error entries of the control vector into blocks.

    The runners name model errors either "q0", "q1", ... (global noise only) or
    "qAT_0", "qR1_0", ... (one block per noise stream), so the block name is
    whatever precedes the time index. Derived from the file rather than from the
    model name, so this works for any of the six models without a registry.
    """

    blocks = {}
    for i, name in enumerate(varis):
        match = re.fullmatch(r"(q[A-Za-z0-9]*?)_?(\d+)", name)
        if match:
            prefix, idx = match.group(1), int(match.group(2))
            blocks.setdefault(prefix, []).append((idx, i))

    # order each block by time index, not by position in the control vector
    return {
        name: [i for _, i in sorted(entries)] for name, entries in sorted(blocks.items())
    }


def main(path, group):
    dt = open_datatree(path)
    ds = dt[group].ds if group in dt else dt.ds

    varis = [str(v) for v in ds.coords["vari"].values]
    blocks = _find_blocks(varis)

    if not blocks:
        print("No model-error blocks found -- is this a deterministic (_nn) run?")
        return 0

    # truth may be stored once per experiment or once per ensemble member. when
    # it is per member, correlate member by member and report the median, which
    # is a much stronger test than correlating against an ensemble-mean truth
    # (averaging truth across members would wash out the very signal we want).
    per_member = "ens_mem" in ds.controls_truth.dims
    n_mem = ds.sizes["ens_mem"]

    print(f"{path}")
    print(f"  group {group} | {n_mem} members | {len(varis)} controls")
    print(f"  truth: {'per member' if per_member else 'single realisation'}\n")

    label = "median r over members" if per_member else "r of posterior mean"
    header = (
        f"  {'block':6s} {'n':>4s} {'std(truth)':>11s} {'std(post)':>10s}  "
        + "".join(f"{f'L={lag:+d}':>9s}" for lag in LAGS)
    )
    print(f"  [{label}]")
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_ok = True
    for name, idxs in blocks.items():
        varis_blk = [varis[i] for i in idxs]

        post = ds.controls.sel(vari=varis_blk).values
        truth = ds.controls_truth.sel(vari=varis_blk).values

        std_truth = float(np.nanstd(truth))
        std_post = float(np.nanstd(post.mean(axis=0)))

        if std_truth < MIN_TRUTH_STD:
            print(
                f"  {name:6s} {len(idxs):4d} {std_truth:11.4g} {std_post:10.4f}  "
                "  SKIPPED: truth is constant, no timing information"
            )
            continue

        if per_member:
            # one correlation per member per lag, then the median across members
            cors = [
                float(
                    np.nanmedian(
                        [_lag_corr(post[m], truth[m], lag) for m in range(n_mem)]
                    )
                )
                for lag in LAGS
            ]
        else:
            cors = [_lag_corr(post.mean(axis=0), truth, lag) for lag in LAGS]

        peak = LAGS[int(np.nanargmax(cors))]
        ok = peak == 0
        all_ok &= ok

        print(
            f"  {name:6s} {len(idxs):4d} {std_truth:11.4f} {std_post:10.4f}  "
            + "".join(f"{c:9.3f}" for c in cors)
            + ("" if ok else f"  <-- PEAK AT LAG {peak:+d}")
        )

    print()
    if all_ok:
        print("  PASS: every block peaks at lag 0")
    else:
        print(
            "  FAIL: at least one block peaks off zero, i.e. the recovered model\n"
            "        errors are assigned to the wrong year. check the model-error\n"
            "        column indices in get_TLM_matrix."
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=str, help="path to a var-assim output .nc")
    parser.add_argument(
        "--group",
        type=str,
        default="2100",
        help="datatree group (assimilation window end year) to test",
    )
    args = parser.parse_args()

    sys.exit(main(args.path, args.group))
