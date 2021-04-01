import numpy as np
from scipy import stats


class Null:
    """
    An identity object, useful as a starting point for accumulators, e.g.::

    total = Null()
    for i in range(10):
        total += SomeClassThatImplementsAddition(i)
    """

    def __iadd__(self, other):
        return other

    def __eq__(self, other):
        return isinstance(other, Null)


def has_table(file_obj, name: str):
    """
    Helper function for determining if a file has a table, works with h5py, pytables, and pandas
    """
    if hasattr(file_obj, "root"):
        return hasattr(file_obj.root, name)
    return name in file_obj.keys()


def get_table(file_obj, name: str):
    """
    Helper function for getting a certian table from a file, works with h5py, pytables, and pandas
    """
    if hasattr(file_obj, "root"):
        return getattr(file_obj.root, name)
    return file_obj[name]


def get_column(table, name: str):
    """
    Helper function getting a column from a table, works with h5py, pytables, and pandas
    """
    if hasattr(table, "cols"):
        return getattr(table.cols, name)[:]
    return table[name]


def get_constant_column(col):
    """
    Helper function which makesure that all of the entries in a column are exactly the same, and returns
    that value.

    This is nescessary because CORSIKA and NuGen store generation surface parameters in every frame and we
    want to verify that they are all the same.
    """
    val = col[0]
    assert np.ndim(val) == 0
    np.array_equal(val, col, equal_nan=True)
    return val


def append_dicts(first, second):  # pragma: no cover
    """
    concat one dict into another
    Not Currently used
    """
    if not first:
        return second
    if not second:
        return first
    out = {}
    assert first.keys() == second.keys()
    for k in first.keys():
        out[k] = np.r_[first[k], second[k]]
    return out


def check_run_counts(table, nfiles):  # pragma: no cover
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
    print("Claimed Runs = {}, Found Runs = {}, {}".format(nfiles, len(runs), status))
    return ret


def check_nfiles(runcol):  # pragma: no cover
    """
    check that the number of jobs in the file is what the user claims they are
    Not Currently used
    """
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

    print("NFiles={}".format(nfiles))
    print("Warning : Guessing the numbers of nugen files based on the number of unique datasets!")
    print("Number of events per file matches a poisson distribution with mean {}".format(rmean))
    print("chi2 / ndof = {} / {} --> p = {}".format(chi2, hist_y.size - 1, pval))
    print("the probability that a file would have zero events is {}".format(pois))
