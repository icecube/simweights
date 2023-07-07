# SPDX-FileCopyrightText: © 2022 the SimWeights contributors
#
# SPDX-License-Identifier: BSD-2-Clause

"""Command line utility to print some basic information about how a file will be weighted."""

import argparse
import sys

import pandas as pd

import simweights


def find_weighter(fileobj: pd.HDFStore, nfiles: float) -> simweights.Weighter:
    """Try to automatically determine which type of file this is and return the correct weighter."""
    try:
        return simweights.CorsikaWeighter(fileobj)
    except RuntimeError:
        pass
    try:
        return simweights.CorsikaWeighter(fileobj, nfiles=nfiles)
    except AttributeError:
        pass
    try:
        return simweights.NuGenWeighter(fileobj, nfiles=nfiles)
    except AttributeError:
        pass
    try:
        return simweights.GenieWeighter(fileobj)
    except AttributeError:
        pass
    mesg = f"Could not find a suitable weighter for file object `{fileobj.filename}`"
    raise RuntimeError(mesg)


def main() -> int:
    """Main function for command line utility."""
    parser = argparse.ArgumentParser(
        "A simple utility to quickly print basic info about how simweights will weight a simulation file",
    )
    parser.add_argument("filename")
    parser.add_argument("-n", "--nfiles", type=float, default=1)
    parser.add_argument("-f", "--flux", default=None)
    parser.add_argument("-w", "--weighter", default="TriggeredCorsikaWeighter")
    args = parser.parse_args()

    with pd.HDFStore(args.filename, "r") as fileobj:
        try:
            wobj = find_weighter(fileobj, args.nfiles)
        except RuntimeError as error:
            sys.stderr.write(str(error) + "\n")
            return -1

        if args.flux:
            try:
                flux_model = getattr(simweights, args.flux)()
            except AttributeError:
                flux_model = None
            if flux_model is None:
                sys.stderr.write(f"Warning: Cannot find flux model {args.flux}\n")

        else:
            flux_model = None

        print(wobj.tostring(flux_model))
    return 0
