.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

Command line Utility
====================

SimWeights comes with a simple command line utility to quickly print basic info about how it will weight a
simulation file. It prints the Generation Surface of the file, a dictionary of the names of the columns it
will use to weight events, the number of events in the file, and the effective area. If a flux model is
provided at the command line it will also print the event rate and livetime. An example for a CORSIKA file
is shown below.

.. code:: shell-session

    $ simweights Level2_IC86.2016_corsika.020789.000000.hdf5 -f GaisserH3a
    < GenerationSurfaceCollection
         PPlus       : N=636650.0 PowerLaw(-2.65 ,30000.0, 1000000.0) NaturalRateCylinder(1200.0, 600.0, 0.00017457020856805793, 1.0)
         He4Nucleus  : N=350000.0 PowerLaw(-2.6 ,30000.0, 1000000.0) NaturalRateCylinder(1200.0, 600.0, 0.00017457020856805793, 1.0)
         N14Nucleus  : N=171100.0 PowerLaw(-2.6 ,30000.0, 1000000.0) NaturalRateCylinder(1200.0, 600.0, 0.00017457020856805793, 1.0)
         Al27Nucleus : N=186650.0 PowerLaw(-2.6 ,30000.0, 1000000.0) NaturalRateCylinder(1200.0, 600.0, 0.00017457020856805793, 1.0)
         Fe56Nucleus : N=155550.0 PowerLaw(-2.6 ,30000.0, 1000000.0) NaturalRateCylinder(1200.0, 600.0, 0.00017457020856805793, 1.0)
    >
    {'energy': ('PolyplopiaPrimary', 'energy'),
     'event_weight': None,
     'pdgid': ('PolyplopiaPrimary', 'type'),
     'zenith': ('PolyplopiaPrimary', 'zenith')}
    Number of Events :    38444
    Effective Area   :   289893 m²
    Using flux model : GaisserH3a
    Event Rate       :  150.131 Hz
    Livetime         :  227.633 s

The complete command line options are show below:

.. literalinclude:: cmdline.txt
    :language: none
