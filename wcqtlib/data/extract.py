"""Utilities for manipulating audio files."""

import argparse
import claudio
import claudio.fileio
import claudio.sox
from joblib import Parallel, delayed
import json
import librosa
import logging
import numpy as np
import os
import pandas
import progressbar
import sys
import wave

import wcqtlib.config as C
import wcqtlib.data.parse
import wcqtlib.common.utils as utils

logger = logging.getLogger(__name__)


def check_valid_audio_files(datasets_df, write_path=None):
    """
    Tries to load every file, and returns a list of any file
    that fails to load.

    Parameters
    ----------
    datasets_df : pandas.DataFrame

    Returns
    -------
    error_list : list of str
    """
    fail_list = []

    with progressbar.ProgressBar(max_value=len(datasets_df)) as progress:
        i = 0
        try:
            for index, row in datasets_df.iterrows():
                audio_file = row["audio_file"]

                try:
                    aobj = claudio.fileio.AudioFile(audio_file, bytedepth=2)
                    if aobj.duration <= .05:
                        fail_list.append(audio_file)
                except AssertionError:
                    fail_list.append((audio_file, "assertion"))
                except EOFError:
                    fail_list.append((audio_file, "EOFError"))
                except wave.Error:
                    fail_list.append((audio_file, "wave.Error"))

                progress.update(i)
                i += 1
        except KeyboardInterrupt:
            print("Cancelled at {}".format(i))

    if write_path:
        with open(write_path, 'w') as fh:
            fh.write("\n".join([str(x) for x in fail_list]))

    return fail_list


def get_onsets(audio, sr, **kwargs):
    # reshape the damn audio so that librosa likes it.
    reshaped_audio = audio.reshape((audio.shape[0],))
    onset_frames = librosa.onset.onset_detect(
        y=reshaped_audio, sr=sr, **kwargs)
    onset_samples = librosa.frames_to_samples(onset_frames)
    return onset_samples


def split_examples(input_audio_path,
                   output_dir,
                   sil_pct_thresh=0.5,
                   min_voicing_duration=0.05,
                   min_silence_duration=1,
                   skip_processing=False,
                   clean_state=True):
    """Takes an audio file, and splits it up into multiple
    audio files, using silence as the delimiter.

    Parameters
    ----------
    input_audio_path : str
        Full path to the audio file to use.

    output_dir : str
        Full path to the folder where you want to place the
        result files. Will be created if it does not exist.

    sil_pct_thresh : float, default=0.5
        Silence threshold as percentage of maximum sample value.

    min_voicing_duration : float, default=0.05
        Minimum amout of time required to be considered non-silent.

    min_silence_duration : float, default=1
        Minimum amout of time require to be considered silent.

    skip_processing : bool, default=False
        If True, attempt to proceed with system state.

    clean_state : bool, default=True
        If True and `skip_processing` is False, clear out any potentially
        conflicting state before running.

    Returns
    -------
    output_files : list of str
        Audio files created in this process.
    """
    original_name = os.path.basename(input_audio_path)
    filebase = utils.filebase(original_name)
    new_output_path = os.path.join(output_dir, original_name)

    # Make sure hte output directory exists
    utils.create_directory(output_dir)

    old_files = [os.path.join(output_dir, x) for x in os.listdir(output_dir)
                 if filebase in x and original_name != x]

    if clean_state and not skip_processing:
        for f in old_files:
            os.remove(f)

    ready_files = []
    # Split the audio files using claudio.sox
    #  [or skip it and check for split files.]
    if skip_processing or claudio.sox.split_along_silence(
                input_audio_path, new_output_path,
                sil_pct_thresh=sil_pct_thresh,
                min_voicing_dur=min_voicing_duration,
                min_silence_dur=min_silence_duration):

        # Sox generates files of the form:
        # original_name001.abc
        # original_name001.xyz
        process_files = [x for x in os.listdir(output_dir)
                         if filebase in x and original_name != x]

        # For each file generated, map it back to the output directory
        for file_name in process_files:
            audio_path = os.path.join(output_dir, file_name)
            try:
                aobj = claudio.fileio.AudioFile(audio_path, bytedepth=2)
                if aobj.duration >= min_voicing_duration:
                    ready_files.append(audio_path)
                else:
                    os.remove(audio_path)
            # TODO: This would be an AssertionError now (claudio problem), it
            # should be a SoxError in the future. HOWEVER, all of this should
            # be the responsibility of `split_along_silence` anyways.
            except AssertionError as derp:
                logger.warning(
                    "Could not open an output of `split_along_silence`: {}\n"
                    "Died with the following error: {}"
                    "".format(audio_path, derp))
                os.remove(audio_path)

    return ready_files


def split_examples_with_count(input_audio_path,
                              output_dir,
                              expected_count,
                              sil_pct_thresh=0.5,
                              min_voicing_duration=0.05,
                              min_silence_duration=1,
                              skip_processing=False,
                              clean_state=True):
    """Takes an audio file, and splits it up into multiple
    audio files, using silence as the delimiter.

    Parameters
    ----------
    input_audio_path : str
        Full path to the audio file to use.

    output_dir : str
        Full path to the folder where you want to place the
        result files. Will be created if it does not exist.

    expected_count : int
        Expected number of clips to be split from the original file.

    sil_pct_thresh : float, default=0.5
        Silence threshold as percentage of maximum sample value.

    min_voicing_duration : float, default=0.05
        Minimum amout of time required to be considered non-silent.

    min_silence_duration : float, default=1
        Minimum amout of time require to be considered silent.

    skip_processing : bool, default=False
        If True, attempt to proceed with system state.

    clean_state : bool, default=True
        If True and `skip_processing` is False, clear out any potentially
        conflicting state before running.

    Returns
    -------
    output_files : list of str, len=`expected_count` or 0
        Audio files created in this process. The list will be empty if it
        the expected number of clips could not be extracted.
    """
    output_files = split_examples(
        input_audio_path, output_dir,
        sil_pct_thresh=sil_pct_thresh,
        min_voicing_duration=min_voicing_duration,
        min_silence_duration=min_silence_duration,
        skip_processing=skip_processing,
        clean_state=clean_state)

    if len(output_files) != expected_count:
        for f in output_files:
            os.remove(f)

    return output_files if len(output_files) == expected_count else list()


def standardize_one(input_audio_path,
                    output_dir=None,
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

    output_dir : str or None
        Path to write updated files to under the same basename. If None,
        overwrites the input file.

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
    output_fname = None
    try:
        aobj = claudio.fileio.AudioFile(input_audio_path, bytedepth=2)
    except AssertionError as e:
        logger.error("Sox may have failed. Input: {}\n Error: {}. Skipping..."
                     .format(input_audio_path, e))
        return None
    except wave.Error as e:
        logger.error(utils.colored(
            "Wave Error; Sox may have failed. Input: {}\n Error: {}."
            " Skipping...".format(input_audio_path, e)), "red")
        return None

    if aobj.duration == 0:
        return None

    if first_onset_start is not None:
        raise NotImplementedError("This done got turned off.")
        # Find the onsets using librosa
        # onset_samples = get_onsets(audio, sr)

        # first_onset_start_samples = first_onset_start * sr
        # actual_first_onset = onset_samples[0]
        # # Pad the beginning with up to onset_start ms of silence
        # onset_difference = first_onset_start_samples - actual_first_onset

        # # Correct the difference by adding or removing samples
        # # from the beginning.
        # if onset_difference > 0:
        #     # In this case, we need to append this many zeros to the start
        #     audio = np.concatenate([
        #         np.zeros([onset_difference, audio.shape[-1]]),
        #         audio])
        #     audio_modified = True
        # elif onset_difference < 0:
        #     audio = audio[np.abs(onset_difference):]
        #     audio_modified = True

    if center_of_mass_alignment:
        raise NotImplementedError("Center of mass not yet implemented.")

    if final_duration < aobj.duration:
        if output_dir:
            utils.create_directory(output_dir)
            output_fname = os.path.join(
                output_dir, os.path.basename(input_audio_path))

        success = claudio.sox.trim(input_audio_path, output_fname, 0,
                                   final_duration)
        if not success:
            logger.error(utils.colored(
                "claudio.sox.trim Failed: {} -- "
                "moving on...".format(input_audio_path), "red"))

    return input_audio_path if output_fname is None else output_fname


def row_to_notes(index, original_audio_path, dataset, instrument, dynamic,
                 extract_path, split_params, skip_processing,
                 max_duration):
    """Extract notes for a dataframe's row.

    Parameters
    ----------
    index : str
        Index of the dataframe row.

    original_audio_path : str
        Path to the full audio file.

    dataset : str
        Name of this row's dataset.

    instrument : str
        Label of this sound file.

    dynamic : str
        Dynamic level for the recording.

    extract_path : str
        Path to extract the data to.

    split_params : dict
        Key-value params to pass off to `split_along_silence`.

    skip_processing : bool
        If True, rebuild from disk.

    max_duration : scalar
        Maximum duration of a given note file.

    Returns
    -------
    results : list of tuples
        New entries for the extracted notes. Each item in the tuple contains
        (primary_index, secondary_index, record).
    """
    output_dir = os.path.join(extract_path, dataset)

    # Get the note files.
    note_count = wcqtlib.data.parse.get_num_notes_from_uiowa_filename(
        original_audio_path)
    if dataset in ['uiowa'] and note_count:
        result_notes = split_examples_with_count(
            original_audio_path, output_dir,
            expected_count=note_count,
            skip_processing=skip_processing, **split_params)
        if not result_notes:
            # Unable to extract the expected number of examples!
            logger.warning(utils.colored(
                "UIOWA file failed to produce the expected number of "
                "examples ({}): {}."
                .format(note_count, original_audio_path), "yellow"))

    elif dataset in ['rwc', 'uiowa']:
        result_notes = split_examples(
            original_audio_path, output_dir,
            skip_processing=skip_processing, **split_params)
    else:  # for philharmonia, just pass it through.
        result_notes = [original_audio_path]

    results = []
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

        record = dict(audio_file=audio_file_path,
                      dataset=dataset,
                      instrument=instrument,
                      dynamic=dynamic)
        # Hierarchical indexing with (parent, new)
        results += [(index,
                     wcqtlib.data.parse.generate_id(dataset, note_file_path),
                     record)]
    return results


def datasets_to_notes(datasets_df, notes_df, extract_path, max_duration=2.0,
                      skip_processing=False, skip_existing=False,
                      bogus_files=None,
                      split_params=None, num_cpus=-1):
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

    notes_df : pandas.DataFrame
        Existing notes_df, if it exists. Otherwise, a blank one.

    extract_path : str
        Path which new note-separated files can be written to.

    max_duration : float
        Max file length in seconds.

    skip_processing : bool or None
        If true, skips the split-on-silence portion of the procedure, and
        just generates the dataframe from existing files.

    skip_existing : bool
        If True, tries to load an existing notes_df, and skips
        processing those data points if they have already been
        processed (if there's an index in the notes_df matching
        one in the datasets_df, and those files exist.)

    bogus_files : str, or None
        If given, filepaths that misbehaved will be written to disk as JSON.

    split_params : dict, or None
        If provided, parameters to be handed off to `split_along_silence`.

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
    split_params = dict(min_voicing_duration=0.1,
                        min_silence_duration=0.5,
                        sil_pct_thresh=0.5) \
        if split_params is None else split_params

    pool = Parallel(n_jobs=num_cpus, verbose=50)
    fx = delayed(row_to_notes)
    kwargs = dict(split_params=split_params, extract_path=extract_path,
                  skip_processing=skip_processing,
                  max_duration=max_duration)

    if skip_existing and not notes_df.empty:
        bidx = []
        # TODO: this could be prettier, but whatevs.
        for (index, row) in datasets_df.iterrows():
            if index in notes_df.index:
                # See if all the files exist
                test_records = notes_df.loc[index]
                if all(map(os.path.exists, test_records["audio_file"])):
                    continue
            bidx.append(index)

        datasets_df = datasets_df.loc[bidx]

    results = pool(fx(index, row.audio_file, row.dataset,
                      row.instrument, row.dynamic, **kwargs)
                   for (index, row) in datasets_df.iterrows())

    for res in results:
        for idx0, idx1, rec in res:
            indexes[0].append(idx0)
            indexes[1].append(idx1)
            records.append(rec)

    return pandas.concat([
        notes_df,
        pandas.DataFrame(records, index=indexes)])


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


def extract_notes(config, skip_processing=False, skip_existing=True):
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

    skip_existing : bool
        If True, tries to load an existing notes_df, and skips
        processing those data points if they have already been
        processed (if there's an index in the notes_df matching
        one in the datasets_df, and those files exist.)

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

    if skip_existing and os.path.exists(notes_df_path):
        notes_df = pandas.read_json(notes_df_path)
    else:
        notes_df = pandas.DataFrame(columns=datasets_df.columns)

    print("Filtering to selected instrument classes.")
    classmap = wcqtlib.data.parse.InstrumentClassMap()
    filtered_df = filter_datasets_on_selected_instruments(
        datasets_df, classmap.allnames)
    # Make sure only valid class names remain in the instrument field.
    print("Normalizing instrument names.")
    filtered_df = wcqtlib.data.parse.normalize_instrument_names(filtered_df)

    print("Loading Notes DataFrame from {} filtered dataset files".format(
        len(filtered_df)))

    notes_df = datasets_to_notes(filtered_df,
                                 notes_df,
                                 output_path,
                                 max_duration=config['extract/max_duration'],
                                 skip_processing=skip_processing,
                                 skip_existing=skip_existing,
                                 bogus_files=config['extract/bogus_files'],
                                 split_params=config['extract/split_params'])

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
