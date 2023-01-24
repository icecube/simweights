.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

How SimWeight Reads Files
=========================

SimWeights is written under assumption that it will be reading hdf5 files written by IceTray's
`hdfwriter <https://docs.icecube.aq/icetray/main/projects/hdfwriter/index.html>`_.
However it can also read any python object with the same structure.

In order to correctly weight Monte Carlo simulation it in necessary to know two classes of quantities:
the generation surface which describes how the individual events were sampled and the sample quantities
which are the values which were actually selected during the generation process.
The generation surface is defined by quantities such as minimum an maximum energy, power-law index,
cylinder height and radius.
These quantities should all be the same for the entire the dataset, or at least every particle in the
entire dataset.
The sample quantities are things like the for energy, particle type, and zenith angle for the primary
particle. They should be distinct for every particle in the sample assuming no oversampling.
Ideally the the generation surface parameters should be stored only once per dataset and only the
sample quantities stored for every event.
However as described below that is not the case.

The design of icetray lends itself to a data model where data is stored for each event.
In many cases, people writing icetray modules find that it is easier to add the
same information to each DAQ frame than it is to add the information at the job or dataset level.
In order to correctly calculate weights for simulation it is necessary to calculate the area of the
generation surface, which is best thought of as information which pertains to the entire job/file.
For experimental data, information which pertains to timescales longer than single events is available in
the GCD frames, however, adding simulation specific information to GCD frames has its own difficulties.
S-Frames were introduced to provide simulation specific information at the job/file level which
significantly simplifies book keeping.
Unfortunately, large amounts of simulation datasets have already been produced with out S-Frames so, it is
still necessary to calculate weights such datasets.

Before S-Frames, both neutrino-generator and corsika-reader would provide information about the generation
surface and other weighting related quantities in an ``I3MapStringDouble`` in the Q-Frame for every event.
Since the generation surface is exactly the same for every event in the dataset, these quantities will be
the same in each Q-Frame.
However, the sampled quantities, such as energy, zenith angle, and particle type are obviously different
for every event.
Neutrino-Generator and corsika-reader store both of these types of quantities in the ``I3MapStringDouble``.

Although, storing this much redundant information is less than ideal, the storage space taken up by this
redundancy is small compared to the overall size of the file.
The real problem was the bookkeeping issue that this storage strategy created.
Because not all simulation jobs execute successfully, and it is not always necessary to use all of the jobs
in a dataset, if one simply combines a bunch of i3 files, there is no way to know how many jobs contributed
to the final event sample.
Since each job is given a unique ``RunId`` in the ``I3EventHeader``, counting the number of unique
``RunID``\ s could in principle work, but at high cut levels there is no guarantee that at least on event
from every job makes it to the final sample.
The solution that most analyzers arrive at is to count the number of i3 files when handed to I3Writer
when booking and to store that number in the filename of the resulting hdf5 file.
This works but is less than ideal.

For neutrino-generator and corsika-reader SimWeights will determine the generation surface from each
particle type and construct the correct weighting object based on that.
To accomplish this it will examine the weighting table (either ``I3MCWeightDict`` or ``CorsikaWeightMap``)
and determine a list of unique particle types.
It verifies that all the columns that define the surface are the same for each particle type.
The number of events per job in included here as ``NEvents``.
The number of files must be passed to :py:func:`simweights.CorsikaWeighter` and
:py:func:`simweights.NuGenWeighter` as the ``nfiles`` parameter which will be multiplied by ``NEvents``
to get the total number of events in the dataset.

For simulation with S-Frames: Triggered CORSIKA, corsika-reader after S-Frames were added, and genie-reader
there is no need for ``nfiles`` as it can be deduced from the number of S-Frames.
In this case SimWeights will determine the generation surface from S-Frames, but it still relies on
objects in the Q-Frame for the sampled quantities.
The table below shows the what tables simweights looks for for each type of simulation.

+--------------------------+---------------------------+---------------------------------------------+
| type                     | S-Frame                   | Q-Frame                                     |
+--------------------------+---------------------------+---------------------------------------------+
| Triggered CORSIKA        | ``I3PrimaryInjectorInfo`` | ``I3CorsikaWeight``                         |
+--------------------------+---------------------------+---------------------------------------------+
| S-Frame CORSIKA          | ``I3CorsikaInfo``         | ``PolyplopiaPrimary``                       |
+--------------------------+---------------------------+---------------------------------------------+
| CORSIKA without S-Frames | none                      | ``CorsikaWeightMap``, ``PolyplopiaPrimary`` |
+--------------------------+---------------------------+---------------------------------------------+
| neutrino-generator       | none                      | ``I3MCWeightDict``                          |
+--------------------------+---------------------------+---------------------------------------------+
| genie-reader             | ``I3GenieInfo``           | ``I3GenieResult``                           |
+--------------------------+---------------------------+---------------------------------------------+
