"""Various utility functions for making plots.

Adam Michael Bauer
9.12.2025
"""

import os
import warnings

from pathlib import Path
from datetime import date

def make_figure_filename(name: str, outdir: Path,
                         ext: str = 'png') -> str:
    """Generate a filename like outdir/yyyy-mm-dd-name.ext.

    Useful for saving figures to disk during analysis.

    Parameters
    ----------
    name: str
        the name of the figure

    ext: str
        the extension (png, svg, pdf, etc.)

    outdir: str
        the output directory (usually /figs/ for me, but could be wherever the
        figure should go)

    Returns
    -------
    filename: str
        figure filename of the form yyyy-mm-dd-name.ext
    """

    today = date.today().strftime('%Y-%m-%d')
    filename = f"{today}-{name}.{ext}"
    if outdir:
        # Since this is under-the-hood, I want to be very thorough, so add a
        # check that raises a warning if the specified outdir doesn't exist
        if os.path.isdir(outdir):
            return os.path.join(outdir, filename)
        else:
            warnings.warn('WARNING: The outdir you specified does not exist.')
            warnings.warn('Making new direcotry at {}'.format(Path(os.getcwd()) / outdir))
            os.makedirs(outdir, exist_ok=True)
            return os.path.join(outdir, filename)

    return filename