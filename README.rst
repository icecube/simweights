.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors
..
.. SPDX-License-Identifier: BSD-2-Clause

|pre-commit.ci| |tests| |docs| |codecov| |LICENSE|

.. |tests| image:: https://github.com/icecube/simweights/actions/workflows/tests.yml/badge.svg?branch=main
  :target: https://github.com/icecube/simweights/actions/workflows/tests.yml
.. |docs| image:: https://github.com/icecube/simweights/actions/workflows/docs.yml/badge.svg?branch=main
  :target: https://docs.icecube.aq/simweights/main
.. |codecov| image:: https://codecov.io/gh/icecube/simweights/branch/main/graph/badge.svg?token=LQ20TAMY9C
  :target: https://codecov.io/gh/icecube/simweights
.. |LICENSE| image:: https://img.shields.io/badge/License-BSD_2--Clause-orange.svg
  :target: https://opensource.org/licenses/BSD-2-Clause
.. |pre-commit.ci| image:: https://results.pre-commit.ci/badge/github/icecube/simweights/main.svg
   :target: https://results.pre-commit.ci/latest/github/icecube/simweights/main
   :alt: pre-commit.ci status

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

| **Required**: `numpy`_, `scipy`_
| **Optional**: `h5py`_, `pytables`_, `pandas`_, `uproot`_, `nuflux`_

.. _numpy: https://numpy.org/
.. _scipy: https://scipy.org/
.. _h5py: https://www.h5py.org/
.. _pytables: https://www.pytables.org/
.. _pandas: https://pandas.pydata.org/
.. _uproot: https://github.com/scikit-hep/uproot5
.. _nuflux: https://github.com/icecube/nuflux

Installation
============

To install with pip:

.. code-block:: shell-session

  pip install git+https://github.com/icecube/simweights.git

On certain installs of python on cvmfs you might get the following error:
``ModuleNotFoundError: No module named 'glob'``. If this happens you can add the
following option ``--no-build-isolation`` to the above command.

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

.. code-block:: pycon

  >>> import simweights, pandas
  >>> simfile = pandas.HDFStore("Level2_IC86.2016_corsika.021889.000000.hdf5", "r")
  >>> flux_model = simweights.GaisserH4a()
  >>> weight_obj = simweights.CorsikaWeighter(simfile)
  >>> weights = weight_obj.get_weights(flux_model)
  >>> print(f"Rate = {weights.sum():5.2f} Hz")
  Rate = 122.84 Hz

The value returned by ``get_weights()`` is the rate of events in Hz

For traditional CORSIKA files made with ``corsika-reader`` you will also use
:code:`simweights.CorsikaWeighter()`, but you need to know the number of ``.i3``
files that contributed to create this hdf5 file and pass it as the ``nfiles`` parameter.

For ``neutrino-generator`` you can use :code:`NuGenWeighter()` which also
requires you to know the number of files.
Flux models from `nuflux <https://github.com/icecube/nuflux>`_ can be used:

.. code-block:: pycon

  >>> import nuflux
  >>> simfile = pandas.HDFStore("Level2_IC86.2016_NuMu.020878.000000.hdf5")
  >>> flux_model = nuflux.makeFlux("CORSIKA_GaisserH3a_QGSJET-II")
  >>> weight_obj = simweights.NuGenWeighter(simfile, nfiles=1)
  >>> weights = weight_obj.get_weights(flux_model)
  >>> print(f"Rate = {weights.sum():5.2e} Hz")
  Rate = 1.41e-02 Hz

To weight a spectrum with a function you can also pass a callable to :code:`get_weights()`

.. code-block:: pycon

  >>> weights = weight_obj.get_weights(lambda energy: 7.2e-8 * energy**-2.2)
  >>> print(f"Rate = {weights.sum():5.2e} Hz")
  Rate = 2.34e-05 Hz

You can also pass flux values as a numpy array with the same length as the sample

.. code-block:: pycon

  >>> fluxes = 7.2e-8 * simfile["I3MCWeightDict"]["PrimaryNeutrinoEnergy"] ** -2.2
  >>> weights = weight_obj.get_weights(fluxes)
  >>> print(f"Rate = {weights.sum():5.2e} Hz")
  Rate = 2.34e-05 Hz

You can also pass a scalar to weight all events with the same flux. Passing
a value of ``1.0`` will result in the well known quantity OneWeight divided
by the number of events.

.. code-block:: pycon

  >>> OneWeight = weight_obj.get_weights(1.0)
  >>> OldOneWeight = simfile["I3MCWeightDict"]["OneWeight"] / (simfile["I3MCWeightDict"]["NEvents"] / 2)
  >>> (OneWeight - OldOneWeight).median()
  0.0

Simulation created with ``genie-reader`` can be weighted with :code:`GenieWeighter()`:

.. code-block:: pycon

  >>> simfile = pandas.HDFStore("genie_reader_NuE.hdf5")
  >>> flux_model = nuflux.makeFlux("IPhonda2014_spl_solmax")
  >>> weight_obj = simweights.GenieWeighter(simfile)
  >>> weights = weight_obj.get_weights(flux_model)
  >>> print(f"Rate = {weights.sum():5.2e} Hz")
  Rate = 3.78e+00 Hz

Also note that these examples use ``pandas``. SimWeights will work equally well with
``pandas``, ``h5py``, or ``pytables``.

Documentation
=============

Full documentation is available on the
`IceCube Documentation Server <https://docs.icecube.aq/simweights/main/>`_.

Getting Help
============

Please direct any questions to ``@kjm`` on the slack channel
`#software <https://icecube-spno.slack.com/channels/software>`_.

.. contrib-break

Contributing
============

See the `contributing guide <CONTRIBUTING.rst>`_
