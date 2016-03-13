"""Utilities for manipulating audio files."""

import argparse
import claudio
import claudio.sox
import librosa
import logging
import numpy as np
import os
import pandas
import progressbar

import wcqtlib.config as C
import wcqtlib.data.parse
import wcqtlib.common.utils as utils

logger = logging.getLogger(__name__)


def get_onsets(audio, sr, **kwargs):
    # reshape the damn audio so that librosa likes it.
    reshaped_audio = audio.reshape((audio.shape[0],))
    onset_frames = librosa.onset.onset_detect(
        y=reshaped_audio, sr=sr, **kwargs)
    onset_samples = librosa.frames_to_samples(onset_frames)
    return onset_samples


def split_examples(input_audio_path,
                   output_dir,
                   skip_processing=False):
    """Takes an audio file, and splits it up into multiple
    audio files, using silence as the delimiter.

    Parameters
    ----------
    input_audio_path : str
        Full path to the audio file to use.

    output_dir : str
        Full path to the folder where you want to place the
        result files. Will be created if it does not exist.

    Returns
    -------
    output_files : List of audio files created in this process.
    """
    original_name = os.path.basename(input_audio_path)
    filebase = utils.filebase(original_name)
    new_output_path = os.path.join(output_dir, original_name)

    # Make sure hte output directory exists
    utils.create_directory(output_dir)

    ready_files = []

    # Split the audio files using claudio.sox
    #  [or skip that and look at existing files.]
    if skip_processing or claudio.sox.split_along_silence(
                input_audio_path, new_output_path):

        # Sox generates files of the form:
        # original_name001.xxx
        # original_name001.xxx
        process_files = [x for x in os.listdir(output_dir) if filebase in x]

        # For each file generated
        for file_name in process_files:
            audio_path = os.path.join(output_dir, file_name)
            ready_files.append(audio_path)

    return ready_files


def standardize_one(input_audio_path,
                    output_path=None,
                    first_onset_start=None,
                    center_of_mass_alignment=False,
                    final_duration=None):
    """Takes a single audio file, and standardizes it based
    on the parameters provided.

    Heads up! Modifies the file in place...

    Parameters
    ----------
    input_audio_path : str
        Full path to the audio file to work with.

    output_path : str or None
        Path to write updated files to. If None, overwrites the
        input file.

    first_onset_start : float or None
        If not None, uses librosa's onset detection to find
        the first onset in the file, and then pads the beginning
        of the file with zeros such that the first onset
        ocurrs at first_onset_start seconds.

        If no onsets are discovered, assumes this is an
        empty file, and returns False.

    center_of_mass_alignment : boolean
        If True, aligns the center of mass of the file to
        be at the center of the sample.

    final_duration : float or None
        If not None, trims the final audio file to final_duration
        seconds.

    Returns
    -------
    output_audio_path : str or None
        Valid full file path if succeeded, or None if failed.
    """
    # Load the audio file
    audio_modified = False
    try:
        audio, sr = claudio.read(input_audio_path, channels=1)
    except AssertionError as e:
        logger.error("Sox may have failed. Input: {}\n Error: {}. Skipping..."
                     .format(input_audio_path, e))
        return False

    if len(audio) == 0:
        return False

    if first_onset_start is not None:
        # Find the onsets using librosa
        onset_samples = get_onsets(audio, sr)

        first_onset_start_samples = first_onset_start * sr
        actual_first_onset = onset_samples[0]
        # Pad the beginning with up to onset_start ms of silence
        onset_difference = first_onset_start_samples - actual_first_onset

        # Correct the difference by adding or removing samples
        # from the beginning.
        if onset_difference > 0:
            # In this case, we need to append this many zeros to the start
            audio = np.concatenate([
                np.zeros([onset_difference, audio.shape[-1]]),
                audio])
            audio_modified = True
        elif onset_difference < 0:
            audio = audio[np.abs(onset_difference):]
            audio_modified = True

    if center_of_mass_alignment:
        raise NotImplementedError("Center of mass not yet implemented.")

    if final_duration:
        final_length_samples = final_duration * sr
        # If this is less than the amount of data we have
        if final_length_samples < len(audio):
            audio = audio[:final_length_samples]
            audio_modified = True
        # Otherwise, just leave it at the current length.

    if audio_modified:
        output_audio_path = input_audio_path
        if (output_path):
            utils.create_directory(output_path)
            output_audio_path = os.path.join(
                output_path, os.path.basename(input_audio_path))

        # save the file back out again.
        claudio.write(output_audio_path, audio, samplerate=sr)

        return output_audio_path
    else:
        return False


def datasets_to_notes(datasets_df, extract_path, max_duration=2.0,
                      skip_processing=False):
    """Take the dataset dataframe created in parse.py
    and extract and standardize separate notes from
    audio files which have multiple notes in them.

    Must have separate behaviors for each dataset, as
    they each have different setups w.r.t the number of notes
    in the file.

    RWC : Each file containes scales of many notes.
        The notes themselves don't seem to be defined in the
        file name.

    UIOWA : Each file contains a few motes from a scale.
        The note range is defined in the filename, but
        does not appear to be consistent.
        Also the space between them is not consistent either.
        Keep an ear out for if the blank space algo works here.

    Philharmonia : These files contain single notes,
        and so are just passed through.

    Parameters
    ----------
    dataset_df : pandas.DataFrame
        Dataframe which defines the locations
        of all input audio files in the dataset and
        their associated instrument classes.

    extract_path : str
        Path which new note-separated files can be written to.

    max_duration : float
        Max file length in seconds.

    skip_processing : bool or None
        If true, skips the split-on-silence portion of the procedure, and
        just generates the dataframe from existing files.

    Returns
    -------
    notes_df : pandas.DataFrame
        Dataframe which points to the extracted
        note files, still pointing to the same
        instrument classes as their parent file.

        Indexed By:
            id : [dataset identifier] + [8 char md5 of filename]
        Columns:
            parent_id : id from "dataset" file.
            audio_file : "split" audio file path.
            dataset : dataset it is from
            instrument : instrument label.
            dynamic : dynamic tag
    """
    # Two arrays for multi/hierarchical indexing.
    indexes = [[], []]
    records = []
    i = 0
    with progressbar.ProgressBar(max_value=len(datasets_df)) as progress:
        for (index, row) in datasets_df.iterrows():
            original_audio_path = row['audio_file']
            dataset = row['dataset']
            output_dir = os.path.join(extract_path, dataset)

            # Get the note files.
            if dataset in ['uiowa', 'rwc']:
                result_notes = split_examples(
                    original_audio_path, output_dir,
                    skip_processing=skip_processing)
            else:  # for philharmonia, just pass it through.
                result_notes = [original_audio_path]

            for note_file_path in result_notes:
                audio_file_path = note_file_path
                # For each note, do standardizing (aka check length)
                if not skip_processing:
                    audio_file_path = standardize_one(
                        note_file_path, output_dir,
                        final_duration=max_duration)
                # If standardizing failed, don't keep this one.
                if audio_file_path is None:
                    continue

                # Hierarchical indexing with (parent, new)
                indexes[0].append(index)
                indexes[1].append(wcqtlib.data.parse.generate_id(
                    dataset, note_file_path))
                records.append(
                    dict(audio_file=audio_file_path,
                         dataset=dataset,
                         instrument=row['instrument'],
                         dynamic=row['dynamic']))
            progress.update(i)
            i += 1

    return pandas.DataFrame(records, index=indexes)


def filter_datasets_on_selected_instruments(datasets_df, selected_instruments):
    """Return a dataframe containing only the entries corresponding
    to the selected instruments.

    Parameters
    ----------
    datasets_df : pandas.DataFrame
        DataFrame containing all of the dataset information.

    selected_instruments : list of str (or None)
        List of strings specifying the instruments to select from.
        (If none, don't filter.)

    Returns
    -------
    filtered_df : pandas.DataFrame
        The datasets_df filtered to contain only the selected
        instruments.
    """
    if not selected_instruments:
        return datasets_df

    return datasets_df[datasets_df["instrument"].isin(selected_instruments)]


def filter_df(unfiltered_df, instrument=None, datasets=[]):
    """Return a view of the features_df looking at only
    the instrument and datasets specified.
    """
    new_df = unfiltered_df.copy()

    if instrument:
        new_df = new_df[new_df["instrument"] == instrument]

    if datasets:
        new_df = new_df[new_df["dataset"].isin(datasets)]

    return new_df


def summarize_notes(notes_df):
    """Print a summary of the classes available in summarize_notes."""
    print("Total Note files generated:", len(notes_df))
    print("Total RWC Notes generated:",
          len(notes_df[notes_df["dataset"] == "rwc"]))
    print("Total UIOWA Notes generated:",
          len(notes_df[notes_df["dataset"] == "uiowa"]))
    print("Total Philharmonia Notes generated:",
          len(notes_df[notes_df["dataset"] == "philharmonia"]))


def extract_notes(config, skip_processing=False):
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
    notes_df_path = os.path.join(output_path,
                                 config["dataframes/notes"])

    print("Running Extraction Process")

    print("Loading Datasets DataFrame")
    datasets_df = pandas.read_json(datasets_df_path)
    print("{} audio files in Datasets.".format(len(datasets_df)))

    print("Filtering to selected instrument classes.")
    classmap = wcqtlib.data.parse.InstrumentClassMap()
    filtered_df = filter_datasets_on_selected_instruments(
        datasets_df, classmap.allnames)
    # Make sure only valid class names remain in the instrument field.
    print("Normalizing instrument names.")
    filtered_df = wcqtlib.data.parse.normalize_instrument_names(filtered_df)

    print("Loading Notes DataFrame from {} filtered dataset files".format(
        len(filtered_df)))

    notes_df = datasets_to_notes(filtered_df, output_path,
                                 max_duration=config['extract/max_duration'],
                                 skip_processing=skip_processing)

    summarize_notes(notes_df)

    # notes_df.to_json(notes_df_path)
    # notes_df.reset_index().to_json(notes_df_path)
    notes_df.to_pickle(notes_df_path)

    try:
        # Try to load it and make sure it worked.
        pandas.read_pickle(notes_df_path)
        print("Created artifact: {}".format(
                utils.colored(notes_df_path, "cyan")))
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
    parser.add_argument("--skip_processing", action="store_true",
                        help="Skip the split-on-silence procedure, and just"
                             " generate the dataframe.")
    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    config = C.Config.from_yaml(args.config_path)
    success = extract_notes(config, args.skip_processing)
    sys.exit(0 if success else 1)
