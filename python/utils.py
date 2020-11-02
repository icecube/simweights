import numpy as np
from scipy import stats

def get_constant_column(col):
    val = col[0]
    assert np.ndim(val)==0    
    assert ((val==col).all())
    return val

def check_nfiles(runcol):
    """
    runcol: and array containing a 
    """

    unique_runs, run_counts = np.unique(runcol, return_counts=True)
    nfiles = unique_runs.size
    
    rmean = run_counts.mean()
    rmin  = run_counts.min()
    rmax  = run_counts.max()

    hist_y,hist_x = np.histogram(run_counts, range=[rmin,rmax+1],bins=rmax-rmin+1)
    hist_x=hist_x[:-1]
    p=stats.poisson.pmf(hist_x,rmean)
    pp=stats.poisson.logpmf(hist_y,p)
    chi2,pval = stats.chisquare(hist_y,p*hist_y.sum())
    pois = stats.poisson.pmf(0,rmean)

    print("NFiles={}".format(nfiles))
    print("Warning : Guessing the numbers of nugen files based on the number of unique datasets!")
    print("Number of events per file matches a poisson distribution with mean {}".format(rmean))
    print("chi2 / ndof = {} / {} --> p = {}".format(chi2,hist_y.size-1,pval))
    print("the probability that a file would have zero events is {}".format(pois))
