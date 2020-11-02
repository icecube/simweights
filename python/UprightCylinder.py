import numpy as np

from . import ZenithBias

class UprightCylinder:
    """
    Acceptance (A * Omega) of an upright cylinder
    """
    def __init__(self, length, radius, cosMin=-1, cosMax=1,
                 zdist=ZenithBias.VolumeCorr):

        self.length  = length
        self.radius  = radius
        self.cosMin  = cosMin
        self.cosMax  = cosMax
        self._side   = 2*self.radius*self.length
        self._cap    = np.pi*self.radius**2
        self.average = self.etendue(cosMin, cosMax)
        self.zbias   = ZenithBias(zdist)

    def __repr__(self):
        return ("{}({},{},{:4.2f},{:4.2f},{})"
                .format(self.__class__.__name__,self.length,self.radius,
                        self.cosMin,self.cosMax,self.zbias.name))
    
    def __eq__(self,other):
        return (self.length == other.length and
                self.radius == other.radius and
                self.cosMin == other.cosMin and
                self.cosMax == other.cosMax and
                self.zbias  == other.zbias)

    def projected_area(self,ct):
        assert np.all(ct >=-1)
        assert np.all(ct <=+1)
        return self._cap*abs(ct) + self._side*np.sqrt(1-ct**2)

    def area_weight(self, ct):
        if self.zbias is ZenithBias.VolumeDet:
            return self.projected_area(ct)
        elif self.zbias is ZenithBias.VolumeCorr:
            return self.average
        else:
            raise Error("Unknown zenith distribution {}".format(self.zbias))
            
    def etendue(self, cosMin=None, cosMax=None):
        """
        Integrate A * d\Omega over the given range of zenith angles
    
        :param cosMin: cosine of the maximum zenith angle
        :param cosMax: cosine of the minimum zenith angle
        :returns: a product of area and solid angle. Divide by
          2*pi*(cosMax-cosMin) to obtain the average projected area in
          this zenith angle range
        """

        if cosMin is None:
            cosMin = self.cosMin
        if cosMax is None:
            cosMax = self.cosMax
                
        assert np.all(cosMin >=-1)
        assert np.all(cosMin <=+1)
        assert np.all(cosMax >=-1)
        assert np.all(cosMax <=+1)                

        return np.pi*(
            self._cap*(cosMax*np.abs(cosMax) -cosMin*np.abs(cosMin))
            + self._side*(np.sqrt(1-cosMax**2)*cosMax - np.arccos(cosMax)
                          -np.sqrt(1-cosMin**2)*cosMin + np.arccos(cosMin))
        )
