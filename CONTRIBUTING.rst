
.. SPDX-FileCopyrightText: © 2022 the SimWeights contributors

.. SPDX-License-Identifier: BSD-2-Clause


Contributing to SimWeights
==========================

Since SimWeights is a free and open-source software project. Please follow the guidelines below if you
would like to contribute to SimWeights.


Use Github Issues
-----------------

Whether you found a bug, you want to request an enhancement, or you're actively developing a new feature,
Github Issues is a great place to keep everyone informed about what you're working on.
Click on the label button to provide more info about your topic.
Every time you make a relevant commit, remember to tag the issue (e.g ``git commit -m 'progress on #12'``),
and when you finish and issue you can close it with a commit too! (e.g ``git commit -m 'Close #12'``)

Create A Pull Requests
----------------------

If you have a change you would like to make to the code please submit a
`pull request <https://github.com/icecube/simweights/pulls>`_ on github.
Please, follow the directions below to install for development and to make sure the pre-commit, unit-tests
documentation and code coverage are correct.
Also make sure that your commits have descriptive messages when you submit the pull request.

Installation
------------

If you want to develop simweights you need to install it with flit.
It is strongly recommended that you setup a virtual environment to develop simweights so that you will not
conflict with other python projects installed on your system. The ``-s`` option will symlink the module
into site-packages rather than copying it, so that you can test changes without reinstalling the module.
If installing in a virtual environment you will not need the ``--user`` option.

::

  pip install flit
  git clone git@github.com:icecube/simweights.git
  cd simweights
  flit install [--user] -s

pre-commit
----------

SimWeights uses `precommit <https://pre-commit.com>`_ which will run a number of checks and automatic
updates on the code before committing it to the repository.
These checks will be run on every pull-request submitted to github and ones that do not pass the checks
will be rejected.

You need to verify that the pre-commit checks will pass before you push or pull-request.
pre-commit should have already been installed as a development requirement when you installed simweights
with flit so there is no need to install it yourself.
However, you will need to install pre-commit hooks in checkout's .git directory.
From the checkout directory ::

  pre-commit install

this will add a script to ``.git/hooks/pre-commit`` which will be run before every commit.
You only need to do this once, after cloning simweights.

To run pre-commit on all files without making a commit you can just run::

  pre-commit run --all

in the base of the simweights directory.

The checks include isort, flake8, pylint, and mypy.
Be sure that all checks pass before submitting a pull request.

Tests
-----

SimWeighs has an extensive unit test suite and uses `pytest <https://docs.pytest.org>`_ as a test runner.
Before submitting any pull-request make sure that all the tests pass by running pytest from the base
directory of simweights.

pytest will print out the code coverage obtained by unit tests when you run it.
If you add functionality make sure that you add additional unit tests to the test suite to cover your new
functionality. SimWeights currently has 100% code coverage, make sure that you keep it that way.

You may have noticed that a lot of tests were skipped when you ran pytests, that is fine.
SimWeights also has an extensive integration test suite, based on production simulation files.
To run this suite you need to download the testdata from ``/data/user/kmeagher/simweights_testdata.tar.gz``.
Once downloaded untar it. You can then run the tests with ::

  SIMWEIGHTS_TESTDATA=dir_you_put_the_file pytest

This will run all the tests including testing the simulation files.

SimWeights also has a tox.ini file for running tox.
This will run the tests on multiple versions of python at the same time.
It can be helpful but it is not necessary before most pull requests.

Documentation
-------------

If you modify the documentation you should test it before submitting a pull request.
You can do this by going to the docs directory and running::

  make html

You can also check that all the hyperlinks are not dead links by running::

  make linkcheck SPHINXOPTS="-W --keep-going"

Checking the CI
---------------

After every pull request github actions will automatically run the tests on a number of python versions and platforms,
you can see the results at the `github actions page <https://github.com/icecube/simweights/actions/workflows/unit_tests.yml>`_.
Also check the `pre-commit results <https://results.pre-commit.ci/latest/github/icecube/simweights/main>`_ and the code
coverage at `codecov <https://app.codecov.io/gh/icecube/simweights>`_. Pull requests will not be accepted unless all of these pass.
