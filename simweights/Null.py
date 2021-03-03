class Null:
    """
    An identity object, useful as a starting point for accumulators, e.g.::

    total = Null()
    for i in range(10):
        total += SomeClassThatImplementsAddition(i)
    """
    def __iadd__(self, other):
        return other
    def __eq__(self, other):
        return isinstance(other, Null) or 0 == other
