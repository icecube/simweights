"""
A collection of cosmic ray flux paramerizations.

This files contains the same Cosmic Ray flux models as :file:`weighting/python/fluxes.py`.
However they have been refactored to:

* Use PDG particle codes natively instad of CORSIKA code
* Use :py:module:`numpy.piecewise` instad of `numexpr`
* Follow :py:module:`numpy` broadcasting rules


"""
from enum import Enum,IntEnum
from numpy import array,broadcast_arrays,equal,meshgrid,piecewise,exp,sqrt

class PDGCode(IntEnum):
    NuE         =         12
    NuEBar      =        -12
    NuMu        =         14
    NuMuBar     =        -14
    NuTau       =         16
    NuTauBar    =        -16
    MuMinus     =         13
    MuPlus      =        -13    
    Gamma       =         22
    PPlus       =       2212
    He4Nucleus  = 1000020040
    Li7Nucleus  = 1000030070
    Be9Nucleus  = 1000040090
    B11Nucleus  = 1000050110
    C12Nucleus  = 1000060120
    N14Nucleus  = 1000070140
    O16Nucleus  = 1000080160
    F19Nucleus  = 1000090190
    Ne20Nucleus = 1000100200
    Na23Nucleus = 1000110230
    Mg24Nucleus = 1000120240
    Al27Nucleus = 1000130270
    Si28Nucleus = 1000140280
    P31Nucleus  = 1000150310
    S32Nucleus  = 1000160320
    Cl35Nucleus = 1000170350
    Ar40Nucleus = 1000180400
    K39Nucleus  = 1000190390
    Ca40Nucleus = 1000200400
    Sc45Nucleus = 1000210450
    Ti48Nucleus = 1000220480
    V51Nucleus  = 1000230510
    Cr52Nucleus = 1000240520
    Mn55Nucleus = 1000250550
    Fe56Nucleus = 1000260560
p = PDGCode

def corsika_to_pdg(cpt):
    ctypes = [14,  402,  703,  904, 1105, 1206, 1407, 1608, 1909, 2010, 2311, 
            2412, 2713, 2814, 3115, 3216, 3517, 4018, 3919, 4020, 4521, 4822, 
            5123, 5224, 5525, 5626]
    ptypes = [2212, 1000020040, 1000030070, 1000040090, 1000050110, 1000060120,
        1000070140, 1000080160, 1000090190, 1000100200, 1000110230, 1000120240, 
        1000130270, 1000140280, 1000150310, 1000160320, 1000170350, 1000180400, 
        1000190390, 1000200400, 1000210450, 1000220480, 1000230510, 1000240520, 
        1000250550, 1000260560]
    cpt=array(cpt)
    return piecewise(cpt,[cpt==t for t in ctypes],ptypes)

class CosmicRayFlux:
    """
    Base class for cosmic ray fluxes that uses `numpy.piecewise` for effient
    mathematical evaluation

    Derived must set `ptypes` to enumerate the particle types in this model.
    :py:func:`_funcs` must be set to a list of functions to be called for each particle 
    type. :py:func:`_condition()` can be overidden if additional piecewise conditions are
    desired.
    """
    def _condition(self,E,pt):
        return [ pt == p for p in self.ptypes ]
    def __call__(self,E,pt):
        E, pt = broadcast_arrays(E, pt)
        pcond = self._condition(E,pt)
        return piecewise(E,pcond,self._funcs)

class Hoerandel(CosmicRayFlux):
    """
    All-particle spectrum (up to iron) after Hoerandel_, as implemented
    in dCORSIKA.

    .. _Hoerandel: http://dx.doi.org/10.1016/S0927-6505(02)00198-6
    """
    ptypes = [ p.PPlus, p.He4Nucleus,  p.Li7Nucleus,  p.Be9Nucleus,  p.B11Nucleus,  p.C12Nucleus, 
        p.N14Nucleus,   p.O16Nucleus,  p.F19Nucleus,  p.Ne20Nucleus, p.Na23Nucleus, p.Mg24Nucleus,
        p.Al27Nucleus,  p.Si28Nucleus, p.P31Nucleus,  p.S32Nucleus,  p.Cl35Nucleus, p.Ar40Nucleus, 
        p.K39Nucleus,   p.Ca40Nucleus, p.Sc45Nucleus, p.Ti48Nucleus, p.V51Nucleus,  p.Cr52Nucleus, 
        p.Mn55Nucleus,  p.Fe56Nucleus ]
    _funcs = [
        lambda E: 11776.445965025136   *E**-2.71*(1+(E/(4.49e6* 1))**1.9)**(-2.1/1.9),
        lambda E:  4749.371132996256   *E**-2.64*(1+(E/(4.49e6* 2))**1.9)**(-2.1/1.9),
        lambda E:    86.7088317618298  *E**-2.54*(1+(E/(4.49e6* 3))**1.9)**(-2.1/1.9),
        lambda E:    84.29044403584496 *E**-2.75*(1+(E/(4.49e6* 4))**1.9)**(-2.1/1.9),
        lambda E:   633.6114770238042  *E**-2.95*(1+(E/(4.49e6* 5))**1.9)**(-2.1/1.9),
        lambda E:  1012.2921411827233  *E**-2.66*(1+(E/(4.49e6* 6))**1.9)**(-2.1/1.9),
        lambda E:   339.6783461252935  *E**-2.72*(1+(E/(4.49e6* 7))**1.9)**(-2.1/1.9),
        lambda E:  1721.4707679448027  *E**-2.68*(1+(E/(4.49e6* 8))**1.9)**(-2.1/1.9),
        lambda E:    38.536639802016566*E**-2.69*(1+(E/(4.49e6* 9))**1.9)**(-2.1/1.9),
        lambda E:   382.61133470722905 *E**-2.64*(1+(E/(4.49e6*10))**1.9)**(-2.1/1.9),
        lambda E:    72.00644098601636 *E**-2.66*(1+(E/(4.49e6*11))**1.9)**(-2.1/1.9),
        lambda E:   666.2427806532402  *E**-2.64*(1+(E/(4.49e6*12))**1.9)**(-2.1/1.9),
        lambda E:   109.82414739246525 *E**-2.66*(1+(E/(4.49e6*13))**1.9)**(-2.1/1.9),
        lambda E:  1415.5104103909828  *E**-2.75*(1+(E/(4.49e6*14))**1.9)**(-2.1/1.9),
        lambda E:    31.722233983367293*E**-2.69*(1+(E/(4.49e6*15))**1.9)**(-2.1/1.9),
        lambda E:   102.29054260257044 *E**-2.55*(1+(E/(4.49e6*16))**1.9)**(-2.1/1.9),
        lambda E:    32.23645896660968 *E**-2.68*(1+(E/(4.49e6*17))**1.9)**(-2.1/1.9),
        lambda E:    69.53545126418338 *E**-2.64*(1+(E/(4.49e6*18))**1.9)**(-2.1/1.9),
        lambda E:    47.77105028396875 *E**-2.65*(1+(E/(4.49e6*19))**1.9)**(-2.1/1.9),
        lambda E:   185.06203553374286 *E**-2.70*(1+(E/(4.49e6*20))**1.9)**(-2.1/1.9),
        lambda E:    25.28561864152123 *E**-2.64*(1+(E/(4.49e6*21))**1.9)**(-2.1/1.9),
        lambda E:    76.39737621929389 *E**-2.61*(1+(E/(4.49e6*22))**1.9)**(-2.1/1.9),
        lambda E:    48.981193059270424*E**-2.63*(1+(E/(4.49e6*23))**1.9)**(-2.1/1.9),
        lambda E:   139.16784695018254 *E**-2.67*(1+(E/(4.49e6*24))**1.9)**(-2.1/1.9),
        lambda E:    32.384244406763116*E**-2.46*(1+(E/(4.49e6*25))**1.9)**(-2.1/1.9),
        lambda E:  1201.2410569254007  *E**-2.59*(1+(E/(4.49e6*26))**1.9)**(-2.1/1.9)]

class Hoerandel5(CosmicRayFlux):
    """
    Hoerandel_ with only 5 components, after Becherini_ et al. (also the same as Arne_ Schoenwald's version)

    .. _Hoerandel: http://dx.doi.org/10.1016/S0927-6505(02)00198-6
    .. _Arne: http://www.ifh.de/~arnes/Forever/Hoerandel_Plots/
    .. _Becherini: http://dx.doi.org/10.1016/j.astropartphys.2005.10.005
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.N14Nucleus, p.Al27Nucleus, p.Fe56Nucleus]
    _funcs = [
        lambda E: 11776.445965025136 *E**-2.71*(1+(E/(4.49e6* 1))**1.9)**(-2.1/1.9),
        lambda E:  4749.371132996256 *E**-2.64*(1+(E/(4.49e6* 2))**1.9)**(-2.1/1.9),
        lambda E:  3552.5893555039243*E**-2.68*(1+(E/(4.49e6* 7))**1.9)**(-2.1/1.9),
        lambda E:  3233.6058556071825*E**-2.67*(1+(E/(4.49e6*13))**1.9)**(-2.1/1.9),
        lambda E:  1197.9991050096223*E**-2.58*(1+(E/(4.49e6*26))**1.9)**(-2.1/1.9)]

class Hoerandel_IT(CosmicRayFlux):
    """
    Modified 5-component Hoerandel spectrum with N and Al replaced by O.
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.O16Nucleus, p.Fe56Nucleus]
    _funcs = [
        lambda E: 11776.445965025136 *E**-2.71*(1+(E/(4.49e6* 1))**1.9)**(-2.1/1.9),
        lambda E:  4749.371132996256 *E**-2.64*(1+(E/(4.49e6* 2))**1.9)**(-2.1/1.9),
        lambda E:  7017.460455316394 *E**-2.68*(1+(E/(4.49e6* 8))**1.9)**(-2.1/1.9),
        lambda E:  1197.9991050096223*E**-2.58*(1+(E/(4.49e6*26))**1.9)**(-2.1/1.9)]

class GaisserHillas(CosmicRayFlux):
    """
    Spectral fits from an `internal report`_ (also on the arXiv) by Tom Gaisser_.

    .. _`internal report`: http://internal.icecube.wisc.edu/reports/details.php?type=report&id=icecube%2F201102004
    .. _Gaisser: http://arxiv.org/abs/1111.6675v2
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.N14Nucleus, p.Al27Nucleus, p.Fe56Nucleus]
    _funcs = [
        lambda E: 7860.0*E**(-2.66)*exp(-E/(4e6* 1)),
        lambda E: 3550.0*E**(-2.58)*exp(-E/(4e6* 2)),
        lambda E: 2200.0*E**(-2.63)*exp(-E/(4e6* 7)),
        lambda E: 1430.0*E**(-2.67)*exp(-E/(4e6*13)),
        lambda E: 2120.0*E**(-2.63)*exp(-E/(4e6*26))]

class GaisserH3a(CosmicRayFlux):
    """
    Spectral fits from an `internal report`_ (also on the arXiv) by `Tom Gaisser`_.

    .. _`internal report`: http://internal.icecube.wisc.edu/reports/details.php?type=report&id=icecube%2F201102004
    .. _`Tom Gaisser`: http://arxiv.org/abs/1111.6675v2

    The model H3a with a mixed extra-galactic population (Fig. 2)
    has all iron at the highest energy and would represent a
    scenario in which the cutoff is not an effect of energy loss
    in propagation over cosmic distances through the CMB but is
    instead just the extra-galactic accelerators reaching their
    highest energy.
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.N14Nucleus, p.Al27Nucleus, p.Fe56Nucleus]
    _funcs = [
        lambda E: 7860.0*E**-2.66*exp(-E/(4e6* 1))+20.0*E**-2.4*exp(-E/(3e7* 1))+1.70*E**-2.4*exp(-E/(2e9*1)),
        lambda E: 3550.0*E**-2.58*exp(-E/(4e6* 2))+20.0*E**-2.4*exp(-E/(3e7* 2))+1.70*E**-2.4*exp(-E/(2e9*2)),
        lambda E: 2200.0*E**-2.63*exp(-E/(4e6* 7))+13.4*E**-2.4*exp(-E/(3e7* 7))+1.14*E**-2.4*exp(-E/(2e9*7)),
        lambda E: 1430.0*E**-2.67*exp(-E/(4e6*13))+13.4*E**-2.4*exp(-E/(3e7*13))+1.14*E**-2.4*exp(-E/(2e9*13)),
        lambda E: 2120.0*E**-2.63*exp(-E/(4e6*26))+13.4*E**-2.4*exp(-E/(3e7*26))+1.14*E**-2.4*exp(-E/(2e9*26))]

class GaisserH4a(CosmicRayFlux):
    """
    Spectral fits from an `internal report`_ (also on the arXiv) by `Tom Gaisser`_.

    .. _`internal report`: http://internal.icecube.wisc.edu/reports/details.php?type=report&id=icecube%2F201102004
    .. _`Tom Gaisser`: http://arxiv.org/abs/1111.6675v2

    In the model H4a, on the other hand, the extra-galactic component
    is assumed to be all protons.
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.N14Nucleus, p.Al27Nucleus, p.Fe56Nucleus]
    _funcs = [
        lambda E: 7860*E**-2.66*exp(-E/(4e6* 1))+20.0*E**-2.4*exp(-E/(3e7*1))+200*E**-2.6*exp(-E/6e10),
        lambda E: 3550*E**-2.58*exp(-E/(4e6* 2))+20.0*E**-2.4*exp(-E/(3e7*2)),
        lambda E: 2200*E**-2.63*exp(-E/(4e6* 7))+13.4*E**-2.4*exp(-E/(3e7*7)),
        lambda E: 1430*E**-2.67*exp(-E/(4e6*13))+13.4*E**-2.4*exp(-E/(3e7*13)),
        lambda E: 2120*E**-2.63*exp(-E/(4e6*26))+13.4*E**-2.4*exp(-E/(3e7*26))]

class GaisserH4a_IT(CosmicRayFlux):
    """
    Variation of Gaisser's H4a flux using only four components.

    *This is not a very physical flux*: The Oxygen group is the sum of H4a's Nitrogen and Aluminum groups.
    This is the flux used as an "a priori" estimate of mass-composition to produce the IceTop-only flux.
    Reference: M. G. Aartsen et al. PHYSICAL REVIEW D 88, 042004 (2013)
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.O16Nucleus, p.Fe56Nucleus]
    _funcs = [
        lambda E: 7860*E**-2.66*exp(-E/(4e6* 1))+20.0*E**-2.4*exp(-E/(3e7*1))+200*E**-2.6*exp(-E/6e10),
        lambda E: 3550*E**-2.58*exp(-E/(4e6* 2))+20.0*E**-2.4*exp(-E/(3e7*2)),
        lambda E: 2200*E**-2.63*exp(-E/(4e6* 7))+13.4*E**-2.4*exp(-E/(3e7*7))
                 +1430*E**-2.67*exp(-E/(4e6*13))+13.4*E**-2.4*exp(-E/(3e7*13)),
        lambda E: 2120*E**-2.63*exp(-E/(4e6*26))+13.4*E**-2.4*exp(-E/(3e7*26))]

class Honda2004(CosmicRayFlux):
    """
    Spectrum used to calculate neutrino fluxes in `Honda et al. (2004)`_ (Table 1, with modification from the text).
    ptypes = [p.PPlus,p.He4Nucleus,p.N14Nucleus,p.Al27Nucleus,p.Fe56Nucleus]

    NB: the E_k notation means energy per nucleon!

    .. _`Honda et al. (2004)`: http://link.aps.org/doi/10.1103/PhysRevD.70.043008
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.N14Nucleus, p.Al27Nucleus, p.Fe56Nucleus]
    _funcs = [
        lambda E: (14900.0    )*(E       + 2.15*exp(-0.21 * sqrt(E)))**(-2.74),
        lambda E: (14900.0    )*(100**(2.71-2.74))*(E/1 + 2.15*exp(-0.21* sqrt(E/1)))**(-2.71),
        lambda E: (600.0/ 4.02)*(E/ 4.02 + 1.25*exp(-0.14 * sqrt(E/ 4.02)))**(-2.64),
        lambda E: (33.2 /14.07)*(E/14.07 + 0.97*exp(-0.01 * sqrt(E/14.07)))**(-2.60),
        lambda E: (34.2 /27.13)*(E/27.13 + 2.14*exp(-0.01 * sqrt(E/27.13)))**(-2.79),
        lambda E: (4.45 /56.26)*(E/56.26 + 3.07*exp(-0.41 * sqrt(E/56.26)))**(-2.68)]
    def _condition(self,E,pt):
        return [(pt==2212)*(E<100),(pt==2212)*(E>=100),
                pt==1000020040,pt==1000070140,pt==1000130270,pt==1000260560]

class TIG1996(CosmicRayFlux):
    """
    Spectrum used to calculate prompt neutrino fluxes in `Enberg et al. (2008)`_ (Eq. 30).
    The parameterization was taken directly from an earlier paper by Thunman_ et al.
    Only the nucleon flux was given, so for simplicity we treat it as a proton-only flux.

    .. _`Enberg et al. (2008)`: http://dx.doi.org/10.1103/PhysRevD.78.043005
    .. _Thunman: http://arxiv.org/abs/hep-ph/9505417
    """
    ptypes = [p.PPlus]
    _funcs = [ lambda E: 1.70e4*E**-2.7, lambda E: 1.74e6*E**-3.0, 0]
    def _condition(self,E,pt):
        return [(pt==2212)*(E<5e6),(pt==2212)*(E>=5e6)]

class GlobalFitGST(CosmicRayFlux):
    """
    Spectral fits_ by Gaisser, Stanev and Tilav.

    .. _fits: http://arxiv.org/pdf/1303.3565v1.pdf
    """
    ptypes = [p.PPlus, p.He4Nucleus, p.N14Nucleus, p.Al27Nucleus, p.Fe56Nucleus]
    _funcs= [
        lambda E: 7000*E**-2.66*exp(-E/1.2e5   )+150*E**-2.4*exp(-E/4e6   )+   14*E**-2.4*exp(-E/1.3e9),
        lambda E: 3200*E**-2.58*exp(-E/1.2e5/2 )+ 65*E**-2.3*exp(-E/4e6/2 ),
        lambda E:  100*E**-2.40*exp(-E/1.2e5/7 )+  6*E**-2.3*exp(-E/4e6/7 ),
        lambda E:  130*E**-2.40*exp(-E/1.2e5/13)+  7*E**-2.3*exp(-E/4e6/13),
        lambda E:   60*E**-2.30*exp(-E/1.2e5/26)+2.3*E**-2.2*exp(-E/4e6/26)+0.025*E**-2.2*exp(-E/1.3e9/26)]

class FixedFractionFlux(CosmicRayFlux):
    """
    Total energy per particle flux flux split among mass groups with a constant fraction.

    By default, this takes as basis the flux from Gaisser H4a summed over all its mass groups,
    then multiplies it by the given fraction. This is a quick way to consider different weightings for systematic checks.
    """
    def __init__(self, fractions={}, basis=GaisserH4a_IT(), normalized=True):
        """
        :param fractions: A dictionary of fractions. They must add up to one and they should correspond to the ptypes in basis

        :type fractions: a dictionary with dataclasses.ParticleType as keys
        """
        self.flux = basis
        f = {int(k):0 for k in basis.ptypes}
        f.update({int(k):v for k,v in fractions.items()})
        self.ptypes = list(f.keys())
        self.fracs = list(f.values())
        if normalized:
            assert(sum(self.fracs)==1.)

    def __call__(self, E, ptype):
        """
        :param E: particle energy in GeV
        :param ptype: particle type code
        :type ptype: int
        """
        E, ptype = broadcast_arrays(E, ptype)
        v = sum(self.flux(E,p) for p in self.ptypes)
        cond = self._condition(E,ptype)
        p = piecewise(E,cond,self.fracs)
        return p*v
