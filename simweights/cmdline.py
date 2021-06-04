import argparse
import sys

import pandas as pd

import simweights


def find_weighter(fileobj, nfiles):
    """
    Try to automatically determine which type of file this is and return the correct weighter
    """
    try:
        return simweights.CorsikaWeighter(fileobj)
    except RuntimeError:
        pass
    try:
        return simweights.CorsikaWeighter(fileobj, nfiles=nfiles)
    except RuntimeError:
        pass
    try:
        return simweights.NuGenWeighter(fileobj, nfiles=nfiles)
    except RuntimeError:
        pass
    raise RuntimeError("could not find a suitable weighter for file object " + str(fileobj))


def main():
    """
    Command line utility to print some basic information about how a file will be weighted
    """
    parser = argparse.ArgumentParser(
        "A simple utility to quickly print basic info about how simweights will weight a simulation file"
    )
    parser.add_argument("filename")
    parser.add_argument("-n", "--nfiles", type=int, default=1)
    parser.add_argument("-f", "--flux", default=None)
    parser.add_argument("-w", "--weighter", default="TriggeredCorsikaWeighter")
    args = parser.parse_args()

    fileobj = pd.HDFStore(args.filename, "r")
    try:
        wobj = find_weighter(fileobj, args.nfiles)
    except RuntimeError as error:
        sys.stderr.write(str(error))
        return -1

    if args.flux:
        try:
            flux_model = getattr(simweights, args.flux)()
        except AttributeError:
            flux_model = None

    else:
        flux_model = None

    print(wobj.tostring(flux_model))
    fileobj.close()
    return 0
