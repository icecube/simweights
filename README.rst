
|tests| |docs| |codecov|


.. |tests| image:: https://github.com/icecube/simweights/actions/workflows/unit_tests.yml/badge.svg?branch=main
  :target: https://github.com/icecube/simweights/actions/workflows/unit_tests.yml
.. |docs| image:: https://github.com/icecube/simweights/actions/workflows/build_docs.yaml/badge.svg?branch=main
  :target: https://docs.icecube.aq/simweights/main
.. |codecov| image:: https://codecov.io/gh/icecube/simweights/branch/main/graph/badge.svg?token=LQ20TAMY9C
  :target: https://codecov.io/gh/icecube/simweights

.. badge-break

==========
SimWeights
==========

Pure python library for calculating the weights of Monte Carlo simulation for IceCube.

SimWeights was designed with goal of calculating weights for IceCube simulation in a way that it
is easy to combine combine datasets with different generation parameters into a single sample.
It was also designed to be a stand alone project which does not depend on IceTray in any way so that it can
be installed easily on laptops. SimWeights gathers all the information it needs form information in the
hdf5 file so there is no need for access to the simulation production database. SimWeights works with
files produced with corsika-reader, neutrino-generator, and genie-reader.

.. doc-break

Prerequisites
=============

| **Required**: ``numpy``, ``scipy``
| **Optional**: ``h5py``, ``pytables``, ``pandas``, ``nuflux``

Installation
============

To install with pip:

.. code-block:: shell-session

  pip install [--user] git+https://github.com/icecube/simweights.git

If you want to develop simweights you can install directly with flit. 
The ``-s`` option will symlink the module into site-packages rather than copying it, 
so that you can test changes without reinstalling the module:

.. code-block:: shell-session

  pip install flit
  git clone git@github.com:icecube/simweights.git
  cd simweights
  flit install [--user] -s

Basic Usage
===========

For triggered CORSIKA or CORSIKA produced by ``corsika-reader`` with S-Frames files use
``CorsikaWeighter()`` without any additional arguments:

.. code-block:: python

  >>> import simweights, pandas
  >>> simfile = pandas.HDFStore('Level2_IC86.2016_corsika.021889.000000.hdf5', 'r')
  >>> flux_model = simweights.GaisserH4a()
  >>> weight_obj = simweights.CorsikaWeighter(simfile)
  >>> weights = weight_obj.get_weights(flux_model)
  >>> print(f'Rate = {weights.sum():5.2f} Hz')
  Rate = 122.84 Hz

The value returned by ``get_weights()`` is the rate of events in Hz

For traditional CORSIKA files made with ``corsika-reader`` you will also use 
:code:`simweights.CorsikaWeighter()`, but you need to know the number of ``.i3`` 
files that contributed to create this hdf5 file and pass it as the ``nfiles`` parameter.

For ``neutrino-generator`` you can use :code:`NuGenWeighter()` which also 
requires you to know the number of files. 
Flux models from `nuflux <https://github.com/icecube/nuflux>`_ can be used:

.. code-block:: python

  >>> import nuflux
  >>> simfile = pandas.HDFStore('Level2_IC86.2016_NuMu.020878.000000.hdf5')
  >>> flux_model = nuflux.makeFlux('CORSIKA_GaisserH3a_QGSJET-II')
  >>> weight_obj = simweights.NuGenWeighter(simfile, nfiles=1)
  >>> weights = weight_obj.get_weights(flux_model)
  >>> print(f'Rate = {weights.sum():5.2e} Hz')
  Rate = 1.41e-02 Hz

Simulation created with ``genie-reader`` can be weighted with :code:`GenieWeighter()`:

.. code-block:: python

  >>> simfile = pandas.HDFStore('genie_reader_NuE_C_corr.hdf5')
  >>> flux_model = nuflux.makeFlux('IPhonda2014_spl_solmax')
  >>> weight_obj = simweights.GenieWeighter(simfile)
  >>> weights = weight_obj.get_weights(flux_model)
  >>> print(f'Rate = {weights.sum():5.2e} Hz')
  Rate = 4.63e-03 Hz

Also note that these examples use ``pandas``. SimWeights will work equally well with
``pandas``, ``h5py``, or ``pytables``.

Documentation
=============

Full documentation is available on the 
`IceCube Documentation Server <https://docs.icecube.aq/simweights/main>`_.

Getting Help
============

Please direct any questions to ``@kjm`` on the slack channel 
`#software <https://icecube-spno.slack.com/channels/software>`_.

Contributing
============

Any bugs or other issues can be filed at the 
`Issue Tracker <https://github.com/icecube/simweights/issues>`_ on GitHub.

Contribution to the source code can done via the 
`Pull Request <https://github.com/icecube/simweights/pulls>`_ tab on GitHub.

.. Contributors

.. License
