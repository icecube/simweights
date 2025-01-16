.. SPDX-FileCopyrightText: © 2025 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

I3Frame Support
===============

You can calculate the weight for a single event directly from an ``I3Frame``
with a number of warnings. First weights are only meaningful for a sample of
Monte Carlo events not a single event. The events calculated here will not be
usfull to combine different samples. Second, the normalization may be off in
subtle ways. For example in triggered CORSIKA there will be 8 S-frames for
each primary type per file which need to be accounted for. When reading an HDF5
file this is correctly accounted for but there is no way to account for this in
IceTray. It is strongly encouraged that you only weight events after obtaining
a complete sample in an HDF5 or similar file. But if you really need to
calculate weights in IceTray you can follow the example below:

.. literalinclude:: ../examples/triggered_corsika_i3file.py
  :start-after: start-example1

Note that the module keeps track of how many S-Frames there are and hence the
factor the weight is incorrect by. But, because of the serial nature of
IceTray, it can't retroactivly apply that correction to events that have
already been processed. The output should look like::

  PPlus       8
  He4Nucleus  8
  N14Nucleus  8
  Al27Nucleus 8
  Fe56Nucleus 8

Indicating that there were 8 S-Frames of each primary type.
