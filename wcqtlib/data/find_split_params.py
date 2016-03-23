"""Utilities for manipulating audio files."""

import argparse
from joblib import Parallel, delayed
import json

import logging
import numpy as np
import os
import pandas
import tempfile
import sys

import wcqtlib.config as C
import wcqtlib.data.parse
import wcqtlib.data.extract as E
import wcqtlib.common.utils as utils


logger = logging.getLogger(__name__)


def check_split_params(audio_path, min_voicing_duration, min_silence_duration,
                       sil_pct_thresh, working_dir=None, clean=True):
    """

    Parameters
    ----------
    audio_path : str
        Path to the full audio file.


    Returns
    -------
    success : bool
        True if splitting seems to have worked, else False.
    """

    output_dir = tempfile.mkdtemp() if working_dir is None else working_dir

    # Get the note files.
    note_count = wcqtlib.data.parse.get_num_notes_from_uiowa_filename(
        audio_path)

    if note_count:
        result_notes = E.split_examples_with_count(
            audio_path, output_dir, expected_count=note_count,
            min_voicing_duration=min_voicing_duration,
            min_silence_duration=min_silence_duration,
            sil_pct_thresh=sil_pct_thresh)
        if not result_notes:
            # Unable to extract the expected number of examples!
            logger.warning(utils.colored(
                "File failed to produce the expected number of "
                "examples ({}): {}."
                .format(note_count, audio_path), "yellow"))

    else:
        result_notes = E.split_examples(
            audio_path, output_dir, min_voicing_duration=min_voicing_duration,
            min_silence_duration=min_silence_duration,
            sil_pct_thresh=sil_pct_thresh)

    if clean:
        for fname in result_notes:
            os.remove(fname)

    return note_count is None or len(result_notes) == note_count


def sweep_parameters(datasets_df, max_attempts=5, num_cpus=-1, seed=None):
    """Take the dataset dataframe created in parse.py
    and

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        Dataframe which defines the locations
        of all input audio files in the dataset and
        their associated instrument classes.

    Returns
    -------

    """
    best_split_params = dict()
    # Back out the index / filepath to a mutable object
    dataset = {index: row.audio_file
               for (index, row) in datasets_df.iterrows()}

    rng = np.random.RandomState(seed=seed)

    logger.info("Starting with {} files.".format(len(dataset)))
    for n in range(max_attempts):
        # split_params = dict(min_voicing_duration=0.1,
        #                     min_silence_duration=0.5,
        #                     sil_pct_thresh=0.5)
        split_params = dict(
            min_voicing_duration=rng.uniform(0.05, 1),
            min_silence_duration=rng.uniform(0.05, 1),
            sil_pct_thresh=rng.uniform(0.1, 0.9))

        pool = Parallel(n_jobs=num_cpus, verbose=50)
        fx = delayed(check_split_params)

        idxs = dataset.keys()
        success = pool(fx(dataset[i], **split_params) for i in idxs)

        # Set params for all that were successful
        best_split_params.update(
            **{i: split_params for i, stat in zip(idxs, success) if stat})

        # Keep the datapoints that weren't
        dataset = {i: dataset[i]
                   for i, stat in zip(idxs, success) if not stat}

        logger.info("After {} iteration(s), {} files remain."
                    "".format(n, len(dataset)))
        # If it's empty, kick on out.
        if not dataset:
            break

        logger.info("Successfully resolved {}/{} files ({:.4})."
                    "".format(len(best_split_params), len(datasets_df),
                              1.*len(best_split_params)/len(datasets_df)))

    return best_split_params


def sweep_dataframe(config, dataset, max_attempts=5):
    """Given a dataframe pointing to dataset files,
    convert the dataset's original files into "note" files,
    containing a single note, and of a maximum duration.

    Parameters
    ----------
    config : config.Config
        The config must specify the following keys:
         * "paths/data_dir" : str
         * "paths/extract_dir" : str
         * "dataframes/datasets" : str
         * "dataframes/notes" : str
         * "extract/max_duration" : float

    skip_processing : bool
        If true, simply examines the notes files already existing,
        and doesn't try to regenerate them. [For debugging only]

    Returns
    -------
    succeeded : bool
        Returns True if the pickle was successfully created,
        and False otherwise.
    """
    output_path = os.path.expanduser(config["paths/extract_dir"])
    datasets_df_path = os.path.join(output_path,
                                    config["dataframes/datasets"])
    split_df_path = os.path.join(output_path,
                                 config["extract/split_params_file"])

    print("Loading Datasets DataFrame")
    datasets_df = pandas.read_json(datasets_df_path)
    print("{} audio files in Datasets.".format(len(datasets_df)))

    print("Filtering to selected instrument classes.")
    classmap = wcqtlib.data.parse.InstrumentClassMap()
    filtered_df = E.filter_datasets_on_selected_instruments(
        datasets_df, classmap.allnames)

    # Make sure only valid class names remain in the instrument field.
    print("Normalizing instrument names.")
    filtered_df = wcqtlib.data.parse.normalize_instrument_names(filtered_df)

    print("Loading Notes DataFrame from {} filtered dataset files".format(
        len(filtered_df)))

    if dataset:
        filtered_df = E.filter_df(filtered_df, datasets=[dataset])
    print("Sweeping over {} files".format(len(filtered_df)))
    split_params = sweep_parameters(filtered_df, 
                                    max_attempts=max_attempts)

    with open(split_df_path, 'w') as fp:
        json.dump(split_params, fp, indent=2)

    try:
        # Try to load it and make sure it worked.
        json.load(open(split_df_path))
        print("Created artifact: {}".format(
                utils.colored(split_df_path, "cyan")))
        return True
    except ValueError:
        logger.warning("Your file failed to save correctly; "
                       "debugging so you can fix it and not have sadness.")
        # If it didn't work, allow us to save it manually
        # TODO: get rid of this? Or not...
        import pdb; pdb.set_trace()
        return False


if __name__ == "__main__":
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                               os.pardir, "data", "master_config.yaml")
    parser = argparse.ArgumentParser(
        description='Use datasets dataframe to generate the notes '
                    'dataframe.')
    parser.add_argument("-c", "--config_path", default=CONFIG_PATH)
    parser.add_argument("--dataset", type=str, default='',
                        help=".")
    parser.add_argument("--max_attempts", type=int, default=5,
                        help=".")
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    config = C.Config.from_yaml(args.config_path)
    success = sweep_dataframe(config, args.dataset, args.max_attempts)
    sys.exit(0 if success else 1)
