import sys
import json
import os

import h5py
import numpy as np

from bubbly.model import ModelGroup
from bubbly.field import get_field
from bubbly.util import chunk, cloud_map

job_file = 'classify_jobs.json'
result_dir = os.path.join('..', 'data', 'full_search')


def field_stamps(lon):
    """
    Return the stamp parameters to classify for each longitude

    Returns
    -------
    A list of lists
    """
    f = get_field(lon)
    stamps = list(f.all_stamps())
    stamps = [s for s in stamps if np.abs(s[1] - lon) <= 0.5]
    return sorted(stamps)


def submit_job(lon):
    """
    Submit a new batch classification job to the cloud

    This also creates or overwrides the appropraite entry
    in classify_jobs.json

    Parameters
    ----------
    lon : longitude to run
    """
    if already_submitted(lon):
        print ("Job already submitted. To force a re-run, "
               "first run\n\t python %s delete %i" % (__file__, lon))
        return

    workers = 100
    stamps = field_stamps(lon)
    model = ModelGroup.load('../models/full_classifier.dat')

    chunks = chunk(stamps, workers)
    jobs = cloud_map(model.decision_function,
                     chunks,
                     return_jobs=True,
                     _label='classify_%3.3i' % lon)
    save_job_ids(lon, jobs)


def delete(lon):
    data = json.load(open(job_file))
    data.pop(str(lon), None)
    with open(job_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def already_submitted(lon):
    data = json.load(open(job_file))
    return str(lon) in data


def retrieve_job(lon):
    """
    Retrieve the results of a previous job submission,
    and save to an hdf5 file

    This creates/overwrites a file at ../data/full_search/<lon>.h5

    Parameters
    ----------
    lon : int. Longitude to retrieve
    """
    import cloud

    jobs = fetch_job_ids(lon)
    stamps = np.array(field_stamps(lon), dtype=np.float32)
    scores = np.hstack(cloud.result(jobs)).astype(np.float32)

    #write to file
    result_file = os.path.join(result_dir, "%3.3i.h5" % lon)
    with h5py.File(result_file, 'w') as f:
        f.create_dataset('stamps', data=stamps, compression=9)
        f.create_dataset('scores', data=scores, compression=9)


def save_job_ids(lon, jobs):
    data = {}
    if os.path.exists(job_file):
        data = json.load(open(job_file))

    data[lon] = [min(jobs), max(jobs)]
    with open(job_file, 'w') as outfile:
        json.dump(data, outfile, indent=2)


def fetch_job_ids(lon):
    err_msg = ("No submitted jobs. "
               "Run python %s submit %i first" % (__file__, lon))
    lon = str(lon)
    if not os.path.exists(job_file):
        raise RuntimeError(err_msg)

    data = json.load(open(job_file))
    if lon not in data:
        raise RuntimeError(err_msg)

    lo, hi = data[lon]
    return range(lo, hi + 1)


def main(argv):
    if len(argv) != 3:
        raise RuntimeError("Usage: \n"
                           "python %s [submit fetch delete] longitude" %
                           __file__)
    lon = int(argv[2])

    if argv[1] == 'submit':
        submit_job(lon)
        sys.exit(0)
    elif argv[1] == 'fetch':
        retrieve_job(lon)
        sys.exit(0)
    elif argv[1] == 'delete':
        delete(lon)
        sys.exit(0)
    else:
        raise RuntimeError("Invalid option: %s" % argv[1])


if __name__ == "__main__":
    main(sys.argv)
