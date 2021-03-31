import numpy as np


class CylinderBase:
    def __init__(self,length,radius,a,b):
        if a < -1 or b > 1:
            raise ValueError(self.__class__.__name__ + ": both a and b must be between -1 and +1")
        if a >= b:
            raise ValueError(self.__class__.__name__ + ": a must be less than b")
        self.length  = length
        self.radius  = radius
        self.a       = a
        self.b       = b
        self._side   = 2 * self.radius * self.length
        self._cap    = np.pi * self.radius**2
        self.etendue = self._diff_etendue(self.b) - self._diff_etendue(self.a)

    def projected_area(self,x):
        assert np.all(x >= -1)
        assert np.all(x <= +1)
        return self._cap * np.abs(x) + self._side * np.sqrt(1 - x**2)

    def _diff_etendue(self,x):
        assert np.all(x >=-1)
        assert np.all(x <=+1)
        return np.pi * (self._cap * x * np.abs(x) + self._side * (x * np.sqrt(1 - x**2) - np.arccos(x)))

    def pdf(self,x):
        raise NotImplementedError()

    def __repr__(self):
        return ("{}({},{},{:4.2f},{:4.2f})".format(self.__class__.__name__, self.length, self.radius, self.a, self.b))

    def __eq__(self,other):
        return (type(self) == type(other) and self.length == other.length and
                self.radius == other.radius and self.a == other.a and self.b == other.b)


#class VolumeDetCylinder(CylinderBase):
    """
    A Cylinder where the primary zenith distribution was uniform and then
    the events were sampled on the cross section perpendicular to the direction
    of the primary. 
    """
#    def _pdf(self, x):
#        return 1 / (2 * np.pi * self.projected_area(x))
    
#    def pdf(self,x):
#        x = np.asfarray(x)
#        return np.piecewise(x, [(x >= self.a) & (x <= self.b)], [self._pdf])


class VolumeCorrCylinder(CylinderBase):
    """
    A Cylinder where the primary zenith distribution was porportional to the 
    cross sectional area a cylinder.
    """
    def __init__(self, length, radius, a, b):
        super().__init__(length, radius, a, b)
        self._N = 1 / self.etendue 

    def pdf(self, x):
        x = np.asfarray(x)
        return np.piecewise(x, [(x >= self.a) & (x <= self.b)], [self._N])
    
