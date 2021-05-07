===================
Creating hdf5 Files
===================

This section discusses how to create files that SimWeights can read.

HDFWriter
=========

The easiest and most common way to do this is to use the icetray module ``I3HDFWriter``. 
`tableio <https://docs.icecube.aq/icetray/main/projects/tableio/index.html>`_ is a set of icetray modules 
for converting  data in ``.i3`` format into tabular format. 
`hdfwriter <https://docs.icecube.aq/icetray/main/projects/hdfwriter/index.html>`_ is a module specifically 
to write hdf5 files. 

.. code:: 

    from icecube import icetray, dataio, dataclasses, simclasses, hdfwriter
    from I3Tray import I3Tray

    tray = I3Tray()
    tray.Add('I3Reader', Filename='dataset_001234.i3.gz')

    tray.Add(hdfwriter.I3HDFWriter,
        Output='dataset_001234.hd5',
        Keys=['I3PrimaryInjectorInfo', 'I3CorsikaWeight' ],
        SubEventStreams=["InIceSplit"]
    )

    tray.Execute()

The above example is for triggered CORSIKA, the keys needed for different types of simulation are as follows:

+--------------------------+---------------------------+---------------------------------------------+
| type                     | S-Frame                   | Q-Frame                                     |
+--------------------------+---------------------------+---------------------------------------------+
| Triggered CORSIKA        | ``I3PrimaryInjectorInfo`` | ``I3CorsikaWeight``                         |
+--------------------------+---------------------------+---------------------------------------------+
| S-Frame CORSIKA          | ``I3CorsikaInfo``         | ``CorsikaWeightMap``, ``PolyplopiaPrimary`` |
+--------------------------+---------------------------+---------------------------------------------+
| CORSIKA without S-Frames | none                      | ``CorsikaWeightMap``, ``PolyplopiaPrimary`` |
+--------------------------+---------------------------+---------------------------------------------+
| neutrino-generator       | none                      | ``I3MCWeightDict``                          |
+--------------------------+---------------------------+---------------------------------------------+

Presumably, in addition to the values listed here you will also want to include additional items to pass to
the Keys parameter such as reconstructions.
Some of the objects used for weighting have their tableio converter defined in simclasses. 
If simclasses is not loaded then tablio will silently fail and not book anything. 




