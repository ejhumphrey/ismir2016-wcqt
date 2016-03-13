#!/usr/bin/env python
"""Compute CQTs for a collection of audio files.

This script writes the output files under the given output directory:

  "/some/audio/file.mp3" maps to "${output_dir}/file.npz"

Sample Call:
$ python audio_to_cqts.py \
    filelist.json \
    ./cqt_arrays \
    --cqt_params=params.json \
    --num_cpus=2
"""
from __future__ import print_function

import argparse
import claudio
from joblib import delayed
from joblib import Parallel
import json
import librosa
import numpy as np
import os
import pandas
import sys
import time

import wcqtlib.common.utils as utils

CQT_PARAMS = dict(
    hop_length=512, fmin=27.5, n_bins=252, bins_per_octave=36, tuning=0.0,
    filter_scale=1, aggregate=None, norm=1, sparsity=0.0, real=False)

AUDIO_PARAMS = dict(samplerate=11025.0, channels=1, bytedepth=2)


def cqt_one(input_file, output_file, cqt_params=None, audio_params=None,
            skip_existing=True):
    """Compute the CQT for a input/output file Pair.

    Parameters
    ----------
    input_file : str
        Audio file to apply the CQT

    output_file : str
        Path to write the output.

    cqt_params : dict, default=None
        Parameters for the CQT function. See `librosa.cqt`.

    audio_params : dict, default=None
        Parameters for reading the audio file. See `claudio.read`.

    skip_existing : bool, default=True
        Skip outputs that exist.

    Returns
    -------
    success : bool
        True if the output file was successfully created.
    """
    input_exists, output_exists = [os.path.exists(f)
                                   for f in (input_file, output_file)]
    if not input_exists:
        print("[{0}] Input file doesn't exist, skipping: {1}"
              "".format(time.asctime(), input_file))
        return input_exists

    if skip_existing and output_exists:
        print("[{0}] Output file exists, skipping: {1}"
              "".format(time.asctime(), output_file))
        return output_exists

    print("[{0}] Starting {1}".format(time.asctime(), input_file))
    if not cqt_params:
        cqt_params = CQT_PARAMS.copy()

    if not audio_params:
        audio_params = AUDIO_PARAMS.copy()

    print("[{0}] Audio conversion {1}".format(time.asctime(), input_file))
    x, fs = claudio.read(input_file, **audio_params)
    print("[{0}] Computing features {1}".format(time.asctime(), input_file))
    cqt_spectra = np.array([np.abs(librosa.cqt(x_c, sr=fs, **cqt_params).T)
                            for x_c in x.T])
    frame_idx = np.arange(cqt_spectra.shape[1])
    time_points = librosa.frames_to_time(
        frame_idx, sr=fs, hop_length=cqt_params['hop_length'])
    print("[{0}] Saving: {1}".format(time.asctime(), output_file))
    np.savez(
        output_file, time_points=time_points,
        cqt=np.abs(cqt_spectra).astype(np.float32))
    print("[{0}] Finished: {1}".format(time.asctime(), output_file))
    return os.path.exists(output_file)


def cqt_many(audio_files, output_files, cqt_params=None, audio_params=None,
             num_cpus=-1, verbose=50, skip_existing=True):
    """Compute CQT representation over a number of audio files.

    Parameters
    ----------
    audio_files : list of str, len=n
        Audio files over which to compute the CQT.

    output_files : list of str, len=n
        File paths for writing outputs.

    cqt_params : dict, default=None
        Parameters to use for CQT computation.

    audio_params : dict, default=None
        Parameters to use for loading the audio file.

    num_cpus : int, default=-1
        Number of parallel threads to use for computation.

    Returns
    -------
    success : bool
        True if all input files were processed successfully.
    """
    pool = Parallel(n_jobs=num_cpus, verbose=50)
    dcqt = delayed(cqt_one)
    pairs = zip(audio_files, output_files)
    return all(pool(dcqt(fin, fout, cqt_params, audio_params)
                    for fin, fout in pairs))


def cqt_from_df(config,
                cqt_params=None, audio_params=None, num_cpus=-1,
                verbose=50, skip_existing=True):
    """Compute CQT representation over audio files referenced by
    a dataframe, and return a new dataframe also containing a column
    referencing the cqt files.

    Parameters
    ----------
    config : config.Config
        The config must specify the following keys:
        extract_path : str
            Folder in the data_root where process files will get dumped
        notes_df_fn : str
            Filename of notes_df in the extract_path.
        features_df_fn : str
            Filename of the features_df in the extract_path.

    cqt_params : dict, default=None
        Parameters to use for CQT computation.

    audio_params : dict, default=None
        Parameters to use for loading the audio file.

    num_cpus : int, default=-1
        Number of parallel threads to use for computation.

    verbose : int
        Passed to cqt_many; for "Parallel"

    skip_existing : bool
        If files exist, don't try to extract them.

    Returns
    -------
    success : bool
        True if all files were processed successfully.
    """
    extract_dir = os.path.expanduser(config["paths/extract_dir"])
    cqt_dir = os.path.join(extract_dir, "cqt")
    utils.create_directory(cqt_dir)
    notes_df_path = os.path.join(extract_dir,
                                 config["dataframes/notes"])
    output_df_path = os.path.join(extract_dir,
                                  config["dataframes/features"])
    # Load the dataframe
    notes_df = pandas.read_pickle(notes_df_path)
    # Clear out any bad values here.
    features_df = notes_df[notes_df["audio_file"] != False]

    def features_path_for_audio(audio_path):
        return os.path.join(cqt_dir,
                            utils.filebase(audio_path) + ".npz")

    audio_paths = features_df["audio_file"].tolist()
    cqt_paths = [features_path_for_audio(x) for x in audio_paths]

    # Create a new column in the new dataframe pointing to these new paths
    features_df["cqt"] = pandas.Series(cqt_paths, index=features_df.index)

    result = cqt_many(audio_paths, cqt_paths, cqt_params, audio_params,
                      num_cpus, verbose, skip_existing)

    # If succeeded, write the new dataframe as a pkl.
    if result:
        features_df.to_pickle(output_df_path)
        print("Created artifact: {}".format(
                utils.colored(output_df_path, "cyan")))
        return True
    else:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument("audio_files",
                        metavar="audio_files", type=str,
                        help="A JSON file with a list of audio filepaths.")
    parser.add_argument("output_directory",
                        metavar="output_directory", type=str,
                        help="Directory to save output arrays.")
    parser.add_argument("--cqt_params",
                        metavar="cqt_params", type=str,
                        default='',
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--audio_params",
                        metavar="audio_params", type=str,
                        default='',
                        help="Path to a JSON file of CQT parameters.")
    parser.add_argument("--num_cpus", type=int,
                        metavar="num_cpus", default=-1,
                        help="Number of CPUs over which to parallelize "
                             "computations.")
    parser.add_argument("--verbose", type=int,
                        metavar="verbose", default=0,
                        help="Verbosity level for joblib.")
    parser.add_argument("--skip_existing", action='store_true',
                        help="If True, will skip existing files.")

    args = parser.parse_args()
    with open(args.audio_files) as fp:
        audio_files = json.load(fp)

    cqt_params = None
    audio_params = None

    cqt_params = json.load(open(args.cqt_params)) if args.cqt_params else None
    audio_params = json.load(open(args.audio_params)) \
        if args.audio_params else None

    output_files = [utils.map_io(fin, args.output_directory)
                    for fin in audio_files]
    success = cqt_many(audio_files, output_files,
                       cqt_params=cqt_params, audio_params=audio_params,
                       num_cpus=args.num_cpus, verbose=args.verbose,
                       skip_existing=args.skip_existing)
    sys.exit(0 if success else 1)
