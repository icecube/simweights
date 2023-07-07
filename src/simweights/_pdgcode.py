# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

from enum import IntEnum


class PDGCode(IntEnum):
    """Enumeration of the PDG particle numbering scheme for cosmic-ray primaries.

    The Particle Data Group (PDG) assigns a unique code to each type of particle. The numbering includes
    all known elementary particles, composite particles, and atomic nuclei. However this enumeration is
    only used for cosmic-ray flux models and is limited to particle types in these models.
    """

    # pylint: disable=invalid-name

    MuMinus = 13
    MuPlus = -13
    NuE = 12
    NuEBar = -12
    NuMu = 14
    NuMuBar = -14
    NuTau = 16
    NuTauBar = -16
    Gamma = 22
    PPlus = 2212
    He4Nucleus = 1000020040
    Li7Nucleus = 1000030070
    Be9Nucleus = 1000040090
    B11Nucleus = 1000050110
    C12Nucleus = 1000060120
    N14Nucleus = 1000070140
    O16Nucleus = 1000080160
    F19Nucleus = 1000090190
    Ne20Nucleus = 1000100200
    Na23Nucleus = 1000110230
    Mg24Nucleus = 1000120240
    Al27Nucleus = 1000130270
    Si28Nucleus = 1000140280
    P31Nucleus = 1000150310
    S32Nucleus = 1000160320
    Cl35Nucleus = 1000170350
    Ar40Nucleus = 1000180400
    K39Nucleus = 1000190390
    Ca40Nucleus = 1000200400
    Sc45Nucleus = 1000210450
    Ti48Nucleus = 1000220480
    V51Nucleus = 1000230510
    Cr52Nucleus = 1000240520
    Mn55Nucleus = 1000250550
    Fe56Nucleus = 1000260560
    Co59Nucleus = 1000270590
    Ni59Nucleus = 1000280590
