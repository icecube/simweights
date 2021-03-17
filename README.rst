==========
SimWeights
==========

Pure python library for weighting IceCube simulation.

Basic Usage
===========

For Dynamic Stack CORSIKA files use ``PrimaryWeighter``::

  import simweights,pandas
  simfile = pandas.HDFStore(hdf5filename,'r')
  flux_model = simweights.GaisserH4a()
  weight_obj = simweights.PrimaryWeighter(simfile)
  weights = weight_obj.get_weights(flux_model)

For traditional CORSIKA files made with ``corsika-reader`` you need to use 
``CorsikaWeighter`` and you need to know the number of i3 files that contributed to create this hdf5 file and pass it as the ``nfiles`` parameter.::

  simfile = pandas.HDFStore(hdf5filename,'r')
  flux_model = simweights.GaisserH4a()
  weight_obj = simweights.CorsikaWeighter(simfile,nfiles=10)
  weights = weight_obj.get_weights(flux_model)
