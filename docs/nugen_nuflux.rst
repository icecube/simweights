How to Use NuFlux Models
========================

Weighting with `nuflux <https://github.com/icecube/nuflux>`_ is very easy with 
SimWeights. All you have to do is pass a nuflux model to :code:`get_weights()` 
just like a callable. SimWeights will detect that it is a nuflux model and do 
the correct thing. 

This example creates a number of nuflux models and compares them with a custom
flux model for the IceCube northern track result.

.. literalinclude:: ../examples/nugen_nuflux.py
     :start-after: start-box1
     :end-before: end-box1

.. figure:: nugen_nuflux.svg