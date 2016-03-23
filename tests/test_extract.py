import claudio
import logging
import os
import pandas
import pytest
import shutil
import numpy as np

import wcqtlib.data.parse
import wcqtlib.data.extract

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

pandas.set_option('display.width', 200)

THIS_PATH = os.path.dirname(__file__)


DATA_ROOT = os.path.expanduser("~/data")
RWC_ROOT = os.path.join(DATA_ROOT, "RWC Instruments")
UIOWA_ROOT = os.path.join(DATA_ROOT, "uiowa")
PHIL_ROOT = os.path.join(DATA_ROOT, "philharmonia")


@pytest.fixture
def testfile(workspace):
    """Copies the "mandolin_trem.mp3" file to
    the workspace so we can mess with it,
    and returns the new path."""
    mando_fn = "mandolin_trem.mp3"
    test_mando = os.path.join(THIS_PATH, mando_fn)
    output_file = os.path.join(workspace, mando_fn)

    shutil.copy(test_mando, output_file)
    return output_file


@pytest.fixture
def uiowa_file(workspace):
    """Copies the UIowa file to the workspace so we can mess with it,
    and returns the new path.
    """
    fname = "BbClar.ff.C4B4.mp3"
    input_file = os.path.join(THIS_PATH, fname)
    output_file = os.path.join(workspace, fname)

    shutil.copy(input_file, output_file)
    return output_file


@pytest.fixture
def datasets_df():
    # First, get the datasets_df with all the original files in it
    datasets_df = wcqtlib.data.parse.load_dataframes(DATA_ROOT)
    assert not datasets_df.empty
    return datasets_df


@pytest.fixture
def filtered_datasets_df(datasets_df):
    classmap = wcqtlib.data.parse.InstrumentClassMap()
    return wcqtlib.data.extract.filter_datasets_on_selected_instruments(
        datasets_df, classmap.allnames)


@pytest.fixture
def rwc_df(filtered_datasets_df):
    return filtered_datasets_df[filtered_datasets_df["dataset"] == "rwc"]


@pytest.fixture
def uiowa_df(filtered_datasets_df):
    return filtered_datasets_df[filtered_datasets_df["dataset"] == "uiowa"]


@pytest.fixture
def philharmonia_df(filtered_datasets_df):
    return filtered_datasets_df[
        filtered_datasets_df["dataset"] == "philharmonia"]


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(PHIL_ROOT),
                             os.path.exists(UIOWA_ROOT),
                             os.path.exists(RWC_ROOT)]),
                    reason="Data not found.")
def test_filter_datasets_on_selected_instruments():
    datasets_df = wcqtlib.data.parse.load_dataframes(DATA_ROOT)

    inst_filter = ["guitar"]
    new_df = wcqtlib.data.extract.filter_datasets_on_selected_instruments(
        datasets_df, inst_filter)
    assert all(new_df["instrument"].unique() == inst_filter)

    inst_filter = ["guitar", "piano"]
    new_df = wcqtlib.data.extract.filter_datasets_on_selected_instruments(
        datasets_df, inst_filter)
    assert all([x in new_df["instrument"].unique() for x in inst_filter])


def test_split_examples_single(testfile, workspace):
    output_files = wcqtlib.data.extract.split_examples(
        testfile, workspace)
    assert all(map(os.path.exists, output_files))
    assert len(output_files) == 1
    assert testfile not in output_files


def test_split_examples_uiowa(uiowa_file, workspace):
    output_files = wcqtlib.data.extract.split_examples(
        uiowa_file, workspace, sil_pct_thresh=0.25,
        min_voicing_duration=0.2, min_silence_duration=1.0)
    assert all(map(os.path.exists, output_files))
    assert len(output_files) == 12
    assert uiowa_file not in output_files


def test_split_examples_with_count(uiowa_file, workspace):
    act_count = 12

    output_files = wcqtlib.data.extract.split_examples_with_count(
        uiowa_file, workspace, act_count)
    assert len(output_files) == act_count
    for fname in output_files:
        assert os.path.exists(fname)

    output_files = wcqtlib.data.extract.split_examples_with_count(
        uiowa_file, workspace, 44)
    assert not output_files
    

def test_standardize_one_final_duration(testfile):
    duration = 1.0
    result = wcqtlib.data.extract.standardize_one(
        testfile, first_onset_start=None,
        final_duration=duration)
    assert result
    assert os.path.exists(result)

    audio, sr = claudio.read(testfile)

    # fudge factor is for mp3 and other encoding which
    #  could add extra time.
    fudge_factor = 1.03
    assert (len(audio) / sr) <= (duration * fudge_factor)


def test_standardize_one_final_duration_toosmall(testfile):
    audio, sr = claudio.read(testfile)

    duration = (len(audio) / sr) * 1.5
    result = wcqtlib.data.extract.standardize_one(
        testfile, first_onset_start=None,
        final_duration=duration)
    assert result is not None and result is not False
    assert os.path.exists(result)
    # If the audio duration is less than druation, it should
    # return the original file
    assert result == testfile


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(RWC_ROOT)]),
                    reason="Data not found.")
@pytest.mark.slowtest
def test_rwc_notes(rwc_df, workspace):
    # Pick four files distributed across the df
    input_df = rwc_df[0:200:50]

    # Now, do the conversion into the notes_df
    notes_df = wcqtlib.data.extract.datasets_to_notes(
        input_df, workspace)
    assert not notes_df.empty

    for (index, row) in input_df.iterrows():
        matching_notes = notes_df.loc[index]
        assert not matching_notes.empty and len(matching_notes) > 1


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(UIOWA_ROOT)]),
                    reason="Data not found.")
@pytest.mark.slowtest
def test_uiowa_notes(uiowa_df, workspace):
    # Pick four files distributed across the df
    input_df = uiowa_df[0:400:50]

    # Now, do the conversion into the notes_df
    notes_df = wcqtlib.data.extract.datasets_to_notes(
        input_df, workspace)
    assert not notes_df.empty

    for (index, row) in input_df.iterrows():
        try:
            matching_notes = notes_df.loc[index]
            assert not matching_notes.empty

            # TODO: not successfully splitting UIOWA files :(
            #  Might need to change the threshold for silence.
            if len(row['note']) >= 1:
                assert len(matching_notes) >= 1
        except KeyError as e:
            logger.error("Index not in notes_df for: {}".format(
                row["audio_file"]))

    # todo... better tests?


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(PHIL_ROOT)]),
                    reason="Data not found.")
@pytest.mark.slowtest
def test_phil_notes(philharmonia_df, workspace):
    input_df = philharmonia_df[0:400:50]

    # Now, do the conversion into the notes_df
    notes_df = wcqtlib.data.extract.datasets_to_notes(
        input_df, workspace)
    assert not notes_df.empty

    for (index, row) in input_df.iterrows():
        matching_notes = notes_df.loc[index]
        assert not matching_notes.empty and len(matching_notes) == 1


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(PHIL_ROOT)]),
                    reason="Data not found.")
@pytest.mark.slowtest
@pytest.mark.runme
def test_datasets_to_notes_existing(rwc_df, workspace):
    input_df = rwc_df[0:400:100]

    # Now, do the conversion into the notes_df
    notes_df = pandas.DataFrame()
    notes_df = wcqtlib.data.extract.datasets_to_notes(
        input_df, notes_df, workspace, skip_existing=False,
        num_cpus=1)
    assert not notes_df.empty

    for (index, row) in input_df.iterrows():
        matching_notes = notes_df.loc[index]
        assert not matching_notes.empty and len(matching_notes) >= 1

    input_df = rwc_df[200:800:100]

    # Now, do the conversion into the notes_df
    notes_df = wcqtlib.data.extract.datasets_to_notes(
        input_df, notes_df, workspace, skip_existing=True,
        num_cpus=1)
    assert not notes_df.empty

    for (index, row) in input_df.iterrows():
        matching_notes = notes_df.loc[index]
        assert not matching_notes.empty and len(matching_notes) >= 1

    # TODO Better tests here?
    # assert len(notes_df.index.unique()) == 6


def test_filter_df(datasets_df):
    # Test filtering on different Instruments
    filtered_df = wcqtlib.data.extract.filter_df(
        datasets_df, instrument="bassoon")
    assert filtered_df["instrument"].unique() == ["bassoon"]

    filtered_df = wcqtlib.data.extract.filter_df(
        datasets_df, instrument="trumpet")
    assert filtered_df["instrument"].unique() == ["trumpet"]

    # Test filtering on different datasets and dataset combinations.
    filtered_df = wcqtlib.data.extract.filter_df(
        datasets_df, datasets=["rwc"])
    assert filtered_df["dataset"].unique() == ["rwc"]

    filtered_df = wcqtlib.data.extract.filter_df(
        datasets_df, datasets=["philharmonia"])
    assert filtered_df["dataset"].unique() == ["philharmonia"]

    filtered_df = wcqtlib.data.extract.filter_df(
        datasets_df, datasets=["philharmonia", "rwc"])
    assert set(filtered_df["dataset"].unique()) == \
           set(["philharmonia", "rwc"])
