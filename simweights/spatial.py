import numpy as np


class CylinderBase:
    """
    Abstract base class for cylinder pdf classes
    """

    def __init__(self, length, radius, cos_zen_min, cos_zen_max):
        if cos_zen_min < -1 or cos_zen_max > 1:
            raise ValueError(
                self.__class__.__name__ + ": both cos_zen_min and cos_zen_max must be between -1 and +1"
            )
        if cos_zen_min >= cos_zen_max:
            raise ValueError(self.__class__.__name__ + ": cos_zen_min must be less than cos_zen_max")
        self.length = length
        self.radius = radius
        self.cos_zen_min = cos_zen_min
        self.cos_zen_max = cos_zen_max
        self._side = 2 * self.radius * self.length
        self._cap = np.pi * self.radius ** 2
        self.etendue = self._diff_etendue(self.cos_zen_max) - self._diff_etendue(self.cos_zen_min)

    def projected_area(self, cos_zen):
        """
        Returns the cross sectional area of a cylinder as seen from the angle described by cos_zen
        """
        assert np.all(cos_zen >= -1)
        assert np.all(cos_zen <= +1)
        return self._cap * np.abs(cos_zen) + self._side * np.sqrt(1 - cos_zen ** 2)

    def _diff_etendue(self, cos_zen):
        assert np.all(cos_zen >= -1)
        assert np.all(cos_zen <= +1)
        return np.pi * (
            self._cap * cos_zen * np.abs(cos_zen)
            + self._side * (cos_zen * np.sqrt(1 - cos_zen ** 2) - np.arccos(cos_zen))
        )

    def pdf(self, cos_zen):
        """
        Returns:
          the probability density function for the given zenith angle.
        """
        raise NotImplementedError()

    def __repr__(self):
        return "{}({}, {}, {}, {})".format(
            self.__class__.__name__, self.length, self.radius, self.cos_zen_min, self.cos_zen_max
        )

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.length == other.length
            and self.radius == other.radius
            and self.cos_zen_min == other.cos_zen_min
            and self.cos_zen_max == other.cos_zen_max
        )


class UniformSolidAngleCylinder(CylinderBase):
    r"""
    A Cylinder where the the angular distribution was sampled as if it were uniform on
    the surface of a sphere. The area of the location surface is porportional to the
    cross section of the cylinder perpendicular to the direction of the primary.

    The Monte Carlo must have been generated with the following zenith angle intensity:

    .. math::

      I \propto \cos\theta

    """

    def _pdf(self, cos_zen):
        return 1 / (2 * np.pi * (self.cos_zen_max - self.cos_zen_min) * self.projected_area(cos_zen))

    def pdf(self, cos_zen):
        cos_zen = np.asfarray(cos_zen)
        return np.piecewise(
            cos_zen, [(cos_zen >= self.cos_zen_min) & (cos_zen <= self.cos_zen_max)], [self._pdf]
        )


class NaturalRateCylinder(CylinderBase):
    r"""
    This is the angular distribution used where the primary zenith distribution matched the natural rate of
    an isotropic source incident on a cylinder.

    For a given zenith angle the intensity of particles thrown was poportional to the cross-sectional area
    perpindicular to the direction of the particle. This is the distribution generated by the icetray class
    ``I3Surfaces::Cylinder`` and is what is used for triggered CORSIKA in ``I3PrimaryInjector``.
    It is also what CORSIKA will generate with the ``VOLUMECORR`` option, when the keyword ``DETCFG`` is set
    to :math:`l/(2r)`.

    The Monte Carlo must have been generated with the following zenith angle intensity:

    .. math::

      I \propto r^2\cdot\pi\cdot\sin\theta\cdot(\cos\theta+2/\pi\cdot l/r\cdot\sin\theta)
    """

    def __init__(self, length, radius, cos_zen_min, cos_zen_max):
        super().__init__(length, radius, cos_zen_min, cos_zen_max)
        self._normalization = 1 / self.etendue

    def pdf(self, cos_zen):
        cos_zen = np.asfarray(cos_zen)
        return np.piecewise(
            cos_zen, [(cos_zen >= self.cos_zen_min) & (cos_zen <= self.cos_zen_max)], [self._normalization]
        )


class CircleInjector:
    """
    This represents the spatial distribution used by older neutrino-genertor versions
    where the particle is injected in a cylinder that is parallel to momentum vector of the primary.
    The etendue is just the area of the circle times the solid angle.
    """

    def __init__(self, radius, cos_zen_min, cos_zen_max):
        self.radius = radius
        self.cos_zen_min = cos_zen_min
        self.cos_zen_max = cos_zen_max
        self._cap = np.pi * self.radius ** 2
        self.etendue = 2 * np.pi * (self.cos_zen_max - self.cos_zen_min) * self._cap
        self._normalization = 1 / self.etendue

    def projected_area(self, cos_zen):
        """
        Returns the cross sectional area of the injection area
        """
        # pylint: disable=unused-argument
        return self._cap

    def pdf(self, cos_zen):
        """
        Returns:
          the probability density function for the given zenith angle.
        """
        cos_zen = np.asfarray(cos_zen)
        return np.piecewise(
            cos_zen, [(cos_zen >= self.cos_zen_min) & (cos_zen <= self.cos_zen_max)], [self._normalization]
        )

    def __repr__(self):
        return "CircleInjector({}, {}, {})".format(self.radius, self.cos_zen_min, self.cos_zen_max)

    def __eq__(self, other):
        return (
            type(self) is type(other)
            and self.radius == other.radius
            and self.cos_zen_min == other.cos_zen_min
            and self.cos_zen_max == other.cos_zen_max
        )