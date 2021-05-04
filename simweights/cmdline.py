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
    parser = argparse.ArgumentParser()
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

    print(wobj.surface)
    print(dir(wobj))
    print(wobj.event_map)
    print("Number of Events   : {:8d}".format(len(wobj.get_weights(1))))
    print("Effective Area     : {:8.6g} m²".format(wobj.effective_area()[0][0]))
    if flux_model:
        weights = wobj.get_weights(flux_model)
        print("Using flux model   : {}".format(flux_model))
        print("Event Rate         : {:8.6g} Hz".format(weights.sum()))
        print("Effective Livetime : {:8.6g} s".format(weights.sum() / (weights ** 2).sum()))

    fileobj.close()
    return 0
