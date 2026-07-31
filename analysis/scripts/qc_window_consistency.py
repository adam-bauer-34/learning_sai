"""QC: are the truth and the prior ensemble consistent across assimilation windows?

Each window in a windowing scheme assimilates a longer stretch of the same
pseudo-reality, so lengthening the window should add data without changing
anything that was already there. Concretely:

  - the true initial conditions, parameters and model errors must be unchanged on
    the span the two windows share
  - ensemble member i must keep the same prior initial conditions and parameters
    in every window, and the same prior model errors on the shared span

Neither held before the draws were moved outside the window loop. Both the truth
and the prior were re-drawn per window, and because the dimension of the draw
grows with the window, re-seeding the RNG did not reproduce the earlier values:
numpy factors the covariance by SVD and that factor depends on the dimension, and
`size=n_ens` fills its standard-normal stream row-major so every member's slice
shifts. Measured on a four-window run, the true global model errors moved by up to
0.85 K on the shared span against a truth std of 0.24 K, and 0 of 100 prior
members were reproduced. See var_assim.model_errors.get_window_prefix_inds.

The comparison is exact rather than approximate: each window's vector is a literal
slice of one full-length draw, so anything other than bit-identical is a failure.

The prior is read from `controls_hist` at iteration 0, which is the optimizer's
starting point and therefore the prior draw itself.

Usage
-----
    python analysis/scripts/qc_window_consistency.py <path to output .nc>

Adam Michael Bauer
UChicago
"""

import argparse
import re
import sys

import numpy as np

from datatree import open_datatree

# entries of the control vector that are model errors, i.e. "q0" / "qAT_0" / "qR1_0"
Q_PATTERN = re.compile(r"(q[A-Za-z0-9]*?)_?(\d+)")


def _split_controls(varis):
    """Separate the control names into the fixed head and the model-error blocks.

    Returns
    -------
    fixed: dict name -> index
        initial conditions and parameters, which every window shares in full

    blocks: dict block name -> dict time index -> control index
        model errors, keyed by time index so windows of different length can be
        matched on the span they share rather than by position
    """

    fixed, blocks = {}, {}
    for i, name in enumerate(varis):
        match = Q_PATTERN.fullmatch(name)
        if match:
            prefix = match.group(1) or "q"
            blocks.setdefault(prefix, {})[int(match.group(2))] = i
        else:
            fixed[name] = i

    return fixed, blocks


def _compare(a, b, label, results):
    """Record whether two arrays are bit-identical, with the worst difference."""

    identical = np.array_equal(a, b)
    worst = 0.0 if identical else float(np.abs(np.asarray(a) - np.asarray(b)).max())
    results.append((label, a.size, identical, worst))

    return identical


def main(path):
    dt = open_datatree(path)

    windows = sorted(
        (k for k in dt.children if str(k).isdigit()), key=lambda k: int(k)
    )
    if len(windows) < 2:
        print(f"{path}\n  only {len(windows)} window(s); nothing to compare.")
        return 0

    varis, truth, prior = {}, {}, {}
    for w in windows:
        ds = dt[w].ds
        varis[w] = [str(v) for v in ds.coords["vari"].values]
        truth[w] = np.asarray(ds.controls_truth.values, dtype=float)
        p = np.asarray(ds.controls_hist.isel(iter=0).values, dtype=float)
        # want (ens_mem, vari)
        prior[w] = p if p.shape[-1] == len(varis[w]) else p.T

    n_mem = prior[windows[0]].shape[0]
    print(f"{path}")
    print(f"  windows: {', '.join(windows)} | {n_mem} members\n")

    base = windows[0]
    fixed_b, blocks_b = _split_controls(varis[base])

    all_ok = True
    for w in windows[1:]:
        fixed_w, blocks_w = _split_controls(varis[w])
        results = []

        # initial conditions and parameters: shared in full
        shared = [n for n in fixed_b if n in fixed_w]
        ib = [fixed_b[n] for n in shared]
        iw = [fixed_w[n] for n in shared]
        _compare(truth[base][ib], truth[w][iw], "truth  ics+params", results)
        _compare(prior[base][:, ib], prior[w][:, iw], "prior  ics+params", results)

        # model errors: matched on time index, so only the shared span
        for blk in sorted(blocks_b):
            if blk not in blocks_w:
                continue
            ts = sorted(set(blocks_b[blk]) & set(blocks_w[blk]))
            ib = [blocks_b[blk][t] for t in ts]
            iw = [blocks_w[blk][t] for t in ts]
            _compare(truth[base][ib], truth[w][iw], f"truth  {blk}", results)
            _compare(prior[base][:, ib], prior[w][:, iw], f"prior  {blk}", results)

        print(f"  {base} vs {w}")
        for label, n, identical, worst in results:
            all_ok &= identical
            print(
                f"    {label:20s} n={n:6d}  "
                f"{'identical' if identical else f'DIFFERS  max|diff|={worst:.3e}'}"
            )
        print()

    if all_ok:
        print("  PASS: truth and prior are consistent across every window")
    else:
        print(
            "  FAIL: something changed between windows. the truth or the prior is\n"
            "        being re-drawn per window rather than sliced from one\n"
            "        full-length draw -- check the runner's pre-loop draw block."
        )

    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=str, help="path to a var-assim output .nc")
    args = parser.parse_args()

    sys.exit(main(args.path))
