import numpy as np

from . import ZenithBias

class UprightCylinder:
    """
    Acceptance (A * Omega) of an upright cylinder
    """
    def __init__(self,
                 length, radius, cosMin=-1, cosMax=1,
                 zdist=ZenithBias.VolumeCorr):

        self.length = length
        self.radius = radius
        self.cosMin = cosMin
        self.cosMax = cosMax
        self._solid_angle = 2*np.pi*(cosMax-cosMin)
        self._side = 2*self.radius*self.length
        self._cap = np.pi*self.radius**2
        self.average = self.etendue(cosMin, cosMax)
        self.zbias=ZenithBias(zdist)

    def __repr__(self):
        return ("{}({},{},{:4.2f},{:4.2f},{})"
                .format(self.__class__.__name__,self.length,self.radius,
                        self.cosMin,self.cosMax,self.zbias.name))
    def __eq__(self,other):
        return (self.length == other.length and
                self.radius == other.radius and
                self.cosMin == other.cosMin and
                self.cosMax == other.cosMax)

    def projected_area(self,ct):
        #assert ct <= +1
        #assert ct >= -1
        return self._solid_angle*(self._cap*abs(ct) + self._side*np.sqrt(1-ct**2))        

    def area_weight(self, ct):
        #assert ct <= +1
        #assert ct >= -1        
        if self.zbias is ZenithBias.VolumeDet:
            return self.projected_area(ct)
        elif self.zbias is ZenithBias.VolumeCorr:
            return self.average
        else:
            raise Error("Unknown zenith distribution {}".format(self.zbias))
        
    @staticmethod
    def _integrate_area(a, b, cap, sides):
        return (cap*(b**2-a**2) + sides*(np.arccos(a) - np.arccos(b) - np.sqrt(1-a**2)*a + np.sqrt(1-b**2)*b))/2.
    
    def etendue(self, cosMin=-1., cosMax=1.):
        """
        Integrate A * d\Omega over the given range of zenith angles
    
        :param cosMin: cosine of the maximum zenith angle
        :param cosMax: cosine of the minimum zenith angle
        :returns: a product of area and solid angle. Divide by
          2*pi*(cosMax-cosMin) to obtain the average projected area in
          this zenith angle range
        """
    
        sides = self._side
        cap = self._cap
        
        if (cosMin >= 0 and cosMax >= 0):
            area = self._integrate_area(cosMin, cosMax, cap, sides)
        elif (cosMin < 0 and cosMax <= 0):
            area = self._integrate_area(-cosMax, -cosMin, cap, sides)
        elif (cosMin < 0 and cosMax > 0):
            area = self._integrate_area(0, -cosMin, cap, sides) \
                 + self._integrate_area(0, cosMax, cap, sides)
        else:
            area = np.nan
            raise ValueError("Can't deal with zenith range [%.1e, %.1e]" % (cosMin, cosMax))
        return 2*np.pi*area

    def average(self):
        pass
