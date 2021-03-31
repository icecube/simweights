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


def has_table(f, name: str):
    if hasattr(f, "root"):
        return hasattr(f.root, name)
    return name in f.keys()


def get_table(f, name: str):
    if hasattr(f, "root"):
        return getattr(f.root, name)
    return f[name]


def get_column(table, name: str):
    if hasattr(table, "cols"):
        return getattr(table.cols, name)[:]
    return table[name]


def get_constant_column(col):
    val = col[0]
    assert np.ndim(val) == 0
    assert (val == col).all()
    return val


def append_dicts(first, second):  # pragma: no cover
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
    runs, _ = np.unique(table.cols.Run[:], return_counts=True)
    # more sophisticated checks go here?
    if len(runs) == nfiles:
        s = "OK"
        ret = True
    else:
        s = "Fail"
        ret = False
    print("Claimed Runs = {}, Found Runs = {}, {}".format(nfiles, len(runs), s))
    return ret


def check_nfiles(runcol):  # pragma: no cover
    """
    runcol: and array containing a
    """

    unique_runs, run_counts = np.unique(runcol, return_counts=True)
    nfiles = unique_runs.size

    rmean = run_counts.mean()
    rmin = run_counts.min()
    rmax = run_counts.max()

    hist_y, hist_x = np.histogram(run_counts, range=[rmin, rmax + 1], bins=rmax - rmin + 1)
    hist_x = hist_x[:-1]
    p = stats.poisson.pmf(hist_x, rmean)
    # pp = stats.poisson.logpmf(hist_y, p)
    chi2, pval = stats.chisquare(hist_y, p * hist_y.sum())
    pois = stats.poisson.pmf(0, rmean)

    print("NFiles={}".format(nfiles))
    print("Warning : Guessing the numbers of nugen files based on the number of unique datasets!")
    print("Number of events per file matches a poisson distribution with mean {}".format(rmean))
    print("chi2 / ndof = {} / {} --> p = {}".format(chi2, hist_y.size - 1, pval))
    print("the probability that a file would have zero events is {}".format(pois))
