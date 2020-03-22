import numpy as np

class PowerLaw:
    def __init__(self,gamma:float,emin:float,emax:float):
        self.gamma=gamma
        self.emin=emin
        self.emax=emax        
    def eval(self,value:float) -> float:
        return np.where( (value > self.emin) & (value < self.emax),
                         value**self.gamma,
                         0)
    def integral(self,a:float,b:float) -> float:
        if self.gamma == -1:
            return np.log(b/a)
        else:
            g=self.gamma+1
            return (b**g-a**g)/g        
    def total_integral(self) -> float:
        return self.integral(self.emin,self.emax)    
    def __repr__(self):
        return "{}({:4.2f},{:6.2e},{:6.2e})".format(self.__class__.__name__,
                                                   self.gamma, self.emin, self.emax)

    def __eq__(self,other):
        return self.gamma==other.gamma and self.emin==other.emin and self.emax==other.emax
