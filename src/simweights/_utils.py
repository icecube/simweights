# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

import numbers
from typing import Any, Optional, Union

import numpy as np
from numpy.random import Generator, RandomState
from numpy.typing import ArrayLike, NDArray

from ._pdgcode import PDGCode

IntNumber = Union[int, np.integer]
GeneratorType = Union[Generator, RandomState]
SeedType = Union[GeneratorType, IntNumber, None]


def has_table(file_obj: Any, name: str) -> bool:
    """Helper function for determining if a file has a table, works with h5py, pytables, and pandas."""
    if hasattr(file_obj, "root"):
        return hasattr(file_obj.root, name)
    return name in file_obj


def get_table(file_obj: Any, name: str) -> Any:
    """Helper function for getting a certain table from a file, works with h5py, pytables, and pandas."""
    if hasattr(file_obj, "root"):
        return getattr(file_obj.root, name)
    return file_obj[name]


def has_column(table: Any, name: str) -> bool:
    """Helper function for determining if a table has a column, works with h5py, pytables, and pandas."""
    if hasattr(table, "cols"):
        return hasattr(table.cols, name)
    try:
        table[name]  # pylint: disable=pointless-statement
        return True  # noqa: TRY300
    except (ValueError, KeyError):
        return False


def get_column(table: Any, name: str) -> NDArray[np.float64]:
    """Helper function getting a column from a table, works with h5py, pytables, and pandas."""
    if hasattr(table, "cols"):
        return np.asfarray(getattr(table.cols, name)[:])
    column = table[name]
    if hasattr(column, "array") and callable(column.array):
        return np.asfarray(column.array(library="np"))
    return np.asfarray(column)


def constcol(table: Any, colname: str, mask: Optional[NDArray[np.bool_]] = None) -> float:
    """Helper function which makes sure that all of the entries in a column are exactly the same.

    This is necessary because CORSIKA and NuGen store generation surface parameters in every frame and we
    want to verify that they are all the same.
    """
    col = get_column(table, colname)
    if mask is not None:
        col = col[mask]
    val = col[0]
    assert np.ndim(val) == 0
    assert (val == col).all()
    return float(val)


def corsika_to_pdg(cid: ArrayLike) -> NDArray[np.float64]:
    """Convert CORSIKA particle code to particle data group (PDG) Monte Carlo numbering scheme.

    Note:
        This function will only convert codes that correspond to
        nuclei needed for the flux models in this module. That includes PPlus (14)
        and He4Nucleus (402) through Fe56Nucleus (5626).

    Args:
        code (array_like): CORSIKA codes

    Returns:
        array_like: PDG codes
    """
    cid = np.asarray(cid, dtype=int)
    corsika_pplus = 14
    corsika_min_nucleus = 100
    corsika_max_nucleus = 9999
    return np.piecewise(
        cid,
        [cid == corsika_pplus, (cid >= corsika_min_nucleus) & (cid <= corsika_max_nucleus)],
        [PDGCode.PPlus, lambda c: 1000000000 + 10000 * (c % 100) + 10 * (c // 100)],
    )


def check_run_counts(table: Any, nfiles: float) -> bool:  # pragma: no cover
    """Check that the number of jobs in the file is what the user claims they are.

    Not Currently used.
    """
    runs, _ = np.unique(table.cols.Run[:], return_counts=True)
    # more sophisticated checks go here?
    if len(runs) == nfiles:
        status = "OK"
        ret = True
    else:
        status = "Fail"
        ret = False
    print(f"Claimed Runs = {nfiles}, Found Runs = {len(runs)}, {status}")
    return ret


def check_random_state(seed: SeedType = None) -> GeneratorType:
    """Turn `seed` into a `numpy.random.Generator` instance.

    Args:
        seed : {None, int, `numpy.random.Generator`, `numpy.random.RandomState`}, optional
            If `seed` is None the `numpy.random.Generator` singleton is used.
            If `seed` is an int, a new ``Generator`` instance is used,
            seeded with `seed`.
            If `seed` is already a ``Generator`` or ``RandomState`` instance then
            that instance is used.


    Returns:
        seed : {`numpy.random.Generator`, `numpy.random.RandomState`}
            Random number generator.
    """
    if seed is None or isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.default_rng(seed)
    if isinstance(seed, (RandomState, Generator)):
        return seed
    mesg = f"{seed!r} cannot be used to seed a numpy.random.Generator instance"
    raise ValueError(mesg)
