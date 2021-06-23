How to Calculate Effective Areas
================================

SimWeights objects have a method :py:func:`simweights.Weighter.effective_area` for calculating effective 
areas. Some example usage is shown below.

The first example will calculate effective area in 4 different zenith bins and 25 energy bins
and plot each zenith bin seperatly.

.. literalinclude :: ../examples/effective_area.py
    :start-after: start-example1
    :end-before: end-example1

.. figure :: effective_area_zenith.png

|

This example demonstrates the use of the mask parameter. 
Effective areas are calculated separately for both NuMu and NuMuBar primaries.
Only a single zenith bin is used.
The third call to ``effective_area()`` has both NuMu and NuMuBar and so the output
is averaged over primary type.

.. literalinclude :: ../examples/effective_area.py
    :start-after: start-example2
    :end-before: end-example2

.. figure :: effective_area_type.png

|

The final example shows a different use of the mask parameter.
Three different filters are used as masks creating three different curves.
Again only a single zenith bin is used.

.. literalinclude :: ../examples/effective_area.py
    :start-after: start-example3
    :end-before: end-example3

.. figure :: effective_area_filters.png