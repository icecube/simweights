import numpy as np

def get_constant_column(col):
    val = col[0]
    assert np.ndim(val)==0    
    assert ((val==col).all())

    return val

