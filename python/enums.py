from enum import Enum,IntEnum

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
    N14Nucleus  = 1000070140
    O16Nucleus  = 1000080160
    Al27Nucleus = 1000130270
    Fe56Nucleus = 1000260560

class ZenithBias(Enum):
   Undefined = -1
   FlatDet    = 0
   VolumeDet  = 1
   VolumeCorr = 2
