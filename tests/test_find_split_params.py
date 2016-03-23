import logging
import os
import pandas
import pytest
import shutil

import wcqtlib.data.parse
import wcqtlib.data.extract
import wcqtlib.data.find_split_params as FSP

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


def test_check_split_params_philharmonia(testfile, workspace):
    assert FSP.check_split_params(testfile, 0.5, 0.5, 0.5, workspace, False)
    # Test that the files are still there?


def test_check_split_params_uiowa(uiowa_file, workspace):
    assert FSP.check_split_params(
        uiowa_file, sil_pct_thresh=0.25, min_voicing_duration=0.2,
        min_silence_duration=1.0)


def test_sweep_parameters(uiowa_df):
    uiowa_df = uiowa_df.iloc[:20]
    best_params = FSP.sweep_parameters(uiowa_df, max_attempts=5, num_cpus=-1,
                                       seed=1234)
    assert best_params
    for idx, params in best_params.items():
        assert idx in uiowa_df.index
        assert FSP.check_split_params(uiowa_df.loc[idx].audio_file, **params)
