from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def has_table(file_obj: Any, name: str) -> bool:
    """
    Helper function for determining if a file has a table, works with h5py, pytables, and pandas
    """
    if hasattr(file_obj, "root"):
        return hasattr(file_obj.root, name)
    return name in file_obj.keys()


def get_table(file_obj: Any, name: str) -> Any:
    """
    Helper function for getting a certian table from a file, works with h5py, pytables, and pandas
    """
    if hasattr(file_obj, "root"):
        return getattr(file_obj.root, name)
    return file_obj[name]


def has_column(table: Any, name: str) -> bool:
    """
    Helper function for determining if a table has a column, works with h5py, pytables, and pandas
    """
    if hasattr(table, "cols"):
        return hasattr(table.cols, name)
    return name in table


def get_column(table: Any, name: str) -> NDArray:
    """
    Helper function getting a column from a table, works with h5py, pytables, and pandas
    """
    if hasattr(table, "cols"):
        return getattr(table.cols, name)[:]
    return np.array(table[name])


def constcol(table: Any, colname: str, mask: NDArray[np.bool_] = None) -> float:
    """
    Helper function which makes sure that all of the entries in a column are exactly the same, and returns
    that value.

    This is necessary because CORSIKA and NuGen store generation surface parameters in every frame and we
    want to verify that they are all the same.
    """
    col = get_column(table, colname)
    if mask is not None:
        col = col[mask]
    val = col[0]
    assert np.ndim(val) == 0
    assert (val == col).all()
    return val


def corsika_to_pdg(cid: ArrayLike) -> NDArray:
    """
    Convert CORSIKA particle code to particle data group (PDG) Monte Carlo
    numbering scheme.

    Note:
        This function will only convert codes that correspond to
        nuclei needed for the flux models in this module. That includes PPlus (14)
        and He4Nucleus (402) through Fe56Nucleus (5626).

    Args:
        code (array_like): CORSIKA codes

    Returns:
        array_like: PDG codes
    """
    cid = np.asarray(cid)
    return np.piecewise(
        cid,
        [cid == 14, (cid >= 100) & (cid <= 9999)],
        [2212, lambda c: 1000000000 + 10000 * (c % 100) + 10 * (c // 100)],
    )


def check_run_counts(table: Any, nfiles: int) -> bool:  # pragma: no cover
    """
    check that the number of jobs in the file is what the user claims they are
    Not Currently used
    """
    runs, _ = np.unique(table.cols.Run[:], return_counts=True)
    # more sophisticated checks go here?
    if len(runs) == nfiles:
        status = "OK"
        ret = True
    else:
        status = "Fail"
        ret = False
    print(f"Claimed Runs = {nfiles}, Found Runs = {len(runs)}, {status}")
    return ret


def check_nfiles(runcol: NDArray):  # pragma: no cover
    # pylint: disable=import-outside-toplevel
    """
    check that the number of jobs in the file is what the user claims they are
    Not Currently used
    """
    from scipy import stats  # type: ignore

    unique_runs, run_counts = np.unique(runcol, return_counts=True)
    nfiles = unique_runs.size

    rmean = run_counts.mean()
    rmin = run_counts.min()
    rmax = run_counts.max()

    hist_y, hist_x = np.histogram(run_counts, range=[rmin, rmax + 1], bins=rmax - rmin + 1)
    hist_x = hist_x[:-1]
    prob = stats.poisson.pmf(hist_x, rmean)
    # pp = stats.poisson.logpmf(hist_y, p)
    chi2, pval = stats.chisquare(hist_y, prob * hist_y.sum())
    pois = stats.poisson.pmf(0, rmean)

    print(f"NFiles={nfiles}")
    print("Warning : Guessing the numbers of nugen files based on the number of unique datasets!")
    print(f"Number of events per file matches a poisson distribution with mean {rmean}")
    print(f"chi2 / ndof = {chi2} / {hist_y.size-1} --> p = {pval}")
    print(f"the probability that a file would have zero events is {pois}")
