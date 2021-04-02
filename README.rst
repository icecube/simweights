==========
SimWeights
==========

Pure python library for calculating the weights of Monte Carlo simulation for IceCube.

.. doc-break

Prerequisites
=============

Required: numpy, scipy
Optional: h5py, pytables, pandas


Installation
============

To install with pip: ::

  git clone git@github.com:icecube/simweights.git
  pip install [--user] ./simweights

If you want to develop simweights you can install directly with flit. 
The ``-s`` option will symlink the module into site-packages rather than copying it, 
so that you can test changes without reinstalling the module: ::

	git clone git@github.com:icecube/simweights.git
	cd simweights
	flit install [--user] -s

Usage
=====

For Dynamic Stack CORSIKA files use ``TriggeredCorsikaWeighter``::

  import simweights,pandas
  simfile = pandas.HDFStore('Level2_IC86.2020_corsika.021111.hdf5','r')
  flux_model = simweights.GaisserH4a()
  weight_obj = simweights.TriggeredCorsikaWeighter(simfile)
  weights = weight_obj.get_weights(flux_model)

For traditional CORSIKA files made with ``corsika-reader`` you need to use 
``CorsikaWeighter`` and you need to know the number of i3 files that contributed to create this hdf5 file and pass it as the ``nfiles`` parameter.::

  simfile = pandas.HDFStore('Level2_IC86.2016_corsika.020789.hdf5','r')
  flux_model = simweights.GaisserH4a()
  weight_obj = simweights.CorsikaWeighter(simfile,nfiles=10)
  weights = weight_obj.get_weights(flux_model)

.. Examples

.. Documentation

Getting Help
============

Please direct any questions to ``@kjm`` on the slack channel `#software <https://icecube-spno.slack.com/channels/software>`_.

.. Contributing

.. Contributors

.. License
