
|Unit Tests|


.. |Unit Tests| image:: https://github.com/icecube/simweights/actions/workflows/unit_tests.yml/badge.svg
   :target:

==========
SimWeights
==========

Pure python library for calculating the weights of Monte Carlo simulation for IceCube.

SimWeights was designed with goal of calculating weights for IceCube simulation in a way that it
is easy to combine combine datasets with different generation parameters into a single sample.
It was also designed to be a stand alone project which does not depend on IceTray in any way so that it can
be installed easily on laptops. SimWeights gathers all the information it needs form information in the
hdf5 file so there is no need for access to the simulation production database.

.. doc-break

Prerequisites
=============

| Required: numpy, scipy
| Optional: h5py, pytables, pandas, nuflux

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

Basic Usage
===========

For Dynamic Stack CORSIKA or CORSIKA produced by ``corsika-reader`` with S-Frames files use
``CorsikaWeighter()`` without any additional arguments::

  import simweights, pandas
  simfile = pandas.HDFStore('Level2_IC86.2020_corsika.021111.hdf5','r')
  flux_model = simweights.GaisserH4a()
  weight_obj = simweights.TriggeredCorsikaWeighter(simfile)
  weights = weight_obj.get_weights(flux_model)

For traditional CORSIKA files made with ``corsika-reader`` you use the same function but you need to know 
the number of ``.i3`` files that contributed to create this hdf5 file and pass it as the ``nfiles`` parameter.::

  simfile = pandas.HDFStore('Level2_IC86.2016_corsika.020789.hdf5','r')
  flux_model = simweights.GaisserH4a()
  weight_obj = simweights.CorsikaWeighter(simfile,nfiles=10)
  weights = weight_obj.get_weights(flux_model)

For ``neutrino-generator`` you can use `NuGenWeighter()` which also requires you to know the number of files.
Flux models from `nuflux <https://github.com/icecube/nuflux>`_ can be used. There is no need to convert the
units, SimWeights will recognize nuflux objects and automatically convert the units.::

  simfile = pandas.HDFStore('Level2_IC86.2016_NuMu.020878.000000.hdf5')
  flux_model = nuflux.makeFlux('CORSIKA_GaisserH3a_QGSJET-II')
  weight_obj=simweights.NuGenWeighter(simfile,nfiles=10)
  weights = weight_obj.get_weights(flux_model)

The value returned by ``get_weights()`` is the rate of events in Hz such that the rate or a simulation
sample can be calculated as::

  print('Rate',weights.sum(),'Hz')

Also not that these examples use ``pandas``. SimWeights will work equally well with
``h5py`` or ``pytables``.

.. Examples

.. Documentation

Getting Help
============

Please direct any questions to ``@kjm`` on the slack channel `#software <https://icecube-spno.slack.com/channels/software>`_.

.. Contributing

.. Contributors

.. License
