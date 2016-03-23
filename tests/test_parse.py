""" Test parse.py functions.

(cbj) I'm not really sure how to test these without the files existing, but
for starters, I'm going to do it by creating dummy hierarchies
in the same format.
"""

import json
import numpy as np
import os
import pandas
import pytest

import wcqtlib.data.parse


DATA_ROOT = os.path.expanduser("~/data")
RWC_ROOT = os.path.join(DATA_ROOT, "RWC Instruments")
UIOWA_ROOT = os.path.join(DATA_ROOT, "uiowa")
PHIL_ROOT = os.path.join(DATA_ROOT, "philharmonia")


@pytest.fixture
def dummy_datasets_df():
    dummy_df = pandas.DataFrame([
        ("foo/bar/stuff.aiff", "uiowa"),
        ("what/who/when.aiff", "philharmonia"),
        ("where/when/whoppens.aiff", "rwc")
        ],
        columns=["audio_file", "dataset"])
    return dummy_df


def __test_df_has_data(df):
    assert not df.empty


def __test(value, expected):
        assert value == expected


def __test_pd_output(pd_output, working_dir, dataset):
    """Make sure all the files in the tree exist"""
    # Check for valid columns
    required_columns = ['audio_file', 'dataset', 'instrument', 'dynamic']
    for column in required_columns:
        assert column in pd_output.columns

    # Check files and per row things.
    for row in pd_output.iterrows():
        assert os.path.exists(row[1]['audio_file'])
        assert row[1]['dataset'] == dataset

    classmap = wcqtlib.data.parse.InstrumentClassMap()

    # Mke sure we have all the selected instruments
    pd_instruments = pd_output["instrument"].unique()
    map_inst = [classmap[x] for x in pd_instruments if classmap[x]]
    instrument_found = np.array([(x in classmap.classnames) for x in map_inst])
    assert all(instrument_found), "Dataset {} is missing: {}".format(
        dataset, instrument_found[instrument_found == 0])


def test_to_basename_df(dummy_datasets_df):
    new_df = wcqtlib.data.parse.to_basename_df(dummy_datasets_df)
    assert sorted(new_df.columns) == ['audio_file', 'dataset', 'dirname']
    for index, row in new_df.iterrows():
        assert new_df.loc[index]['audio_file'] == \
            os.path.basename(dummy_datasets_df.loc[index]['audio_file'])


def test_generate_canonical_files(workspace, dummy_datasets_df):
    """Create a dummy dataframe, and make sure it got written out
    as a json file correctly."""
    destination = os.path.join(workspace, "canonical_files.json")
    success = wcqtlib.data.parse.generate_canonical_files(
        dummy_datasets_df, destination)

    assert success
    assert os.path.exists(destination)

    # Try to load it as a dataframe
    reloaded_df = wcqtlib.data.parse.load_canonical_files(destination)
    assert len(reloaded_df) == len(dummy_datasets_df)


def test_rwc_instrument_code_to_name():
    # We'll just test a few.
    test_pairs = [("AS", "saxophone-alto"),
                  ("FG", "bassoon"),
                  ("TB", "trombone"),
                  # No valid mapping should produce itself
                  ("SZ", "SZ"),
                  ("what", "what")]

    for value, expected in test_pairs:
        result = wcqtlib.data.parse.rwc_instrument_code_to_name(value)
        yield __test, result, expected


def test_parse_rwc_path():
    test_pairs = [("011PFNOF.flac", ("piano", "NO", "F")),
                  ("232TUNOF.flac", ("tuba", "NO", "F")),
                  ("472TNA1F", ("TN", "A1", "F"))]

    for value, expected in test_pairs:
        result = wcqtlib.data.parse.parse_rwc_path(value)
        yield __test, result, expected


def test_parse_uiowa_path():
    test_pairs = [("Horn.ff.Bb1B1.aiff", ("Horn", "ff", "Bb1B1")),
                  ("Cello.arco.ff.sulA.C4B4.aiff", ("Cello", "ff", "C4B4")),
                  ("Viola.arco.sulA.ff.A4B4.aiff", ("Viola", "ff", "A4B4")),
                  ("Rubber.aif", ("Rubber", None, None)),
                  ("Guitar.ff.sul_E.C5Bb5.stereo.aif",
                   ("Guitar", "ff", "C5Bb5")),
                  ("Piano.ff.B3.aiff", ("Piano", "ff", "B3")),
                  ("Trumpet.vib.ff.E3B3.aiff", ("Trumpet", "ff", "E3B3"))]

    for value, expected in test_pairs:
        result = wcqtlib.data.parse.parse_uiowa_path(value)
        yield __test, result, expected


def test_parse_num_notes_from_path():
    test_pairs = [("Horn.ff.Bb1B1.aiff", 2),
                  ("Cello.arco.ff.sulA.C4B4.aiff", 12),
                  ("Viola.arco.sulA.ff.A4B4.aiff", 3),
                  ("Rubber.aif", None),
                  ("Guitar.ff.sul_E.C5Bb5.stereo.aif", 11),
                  ("Piano.ff.B3.aiff", 1),
                  ("Trumpet.vib.ff.E3B3.aiff", 8)]

    for value, expected in test_pairs:
        result = wcqtlib.data.parse.get_num_notes_from_uiowa_filename(value)
        yield __test, result, expected


def test_get_note_distance():
    test_pairs = [(('Bb1', 'B1'), 1),
                  (('C4', 'B4'), 11),
                  (('Bb1', 'B2'), 13),
                  (('C3', 'C4'), 12),
                  (('F5', 'F5'), 0),
                  (('C#3', 'C4'), 11)]

    for value, expected in test_pairs:
        result = wcqtlib.data.parse.get_note_distance(value)
        yield __test, result, expected


def test_parse_phil_path():
    test_pairs = [("banjo_B3_very-long_piano_normal.mp3",
                   ("banjo", "B3", "very-long", "piano", "normal")),
                  ("cello_A3_1_fortissimo_arco-normal.mp3",
                   ("cello", "A3", "1", "fortissimo", "arco-normal")),
                  ("trumpet_A3_15_pianissimo_normal.mp3",
                   ("trumpet", "A3", "15", "pianissimo", "normal")),
                  ("double-bass_A1_1_mezzo-forte_arco-normal",
                   ("double-bass", "A1", "1", "mezzo-forte", "arco-normal")),
                  ("/Users/cjacoby/data/philharmonia/www.philharmonia.co.uk/"
                   "assets/audio/samples/double bass/double bass"
                   "/double-bass_E1_phrase_mezzo-forte_arco-au-talon.mp3",
                   ("double-bass", "E1", "phrase", "mezzo-forte",
                    "arco-au-talon"))]

    for value, expected in test_pairs:
        result = wcqtlib.data.parse.parse_phil_path(value)
        yield __test, result, expected


def test_generate_id():
    def __test_hash(dataset, result):
        assert result[0] == dataset[0] and result[0] in ['r', 'u', 'p']
        assert len(result) == 9

    tests = [("rwc", "foobar.mp3"),
             ("philharmonia", "testwhat.foo"),
             ("uiowa", "i'matestfile.aiff")]

    for dataset, filename in tests:
        result = wcqtlib.data.parse.generate_id(dataset, filename)
        yield __test_hash, dataset, result


@pytest.mark.skipif(not os.path.exists(RWC_ROOT),
                    reason="Data not found.")
def test_rwc_to_dataframe():
    """Test that an input folder with files in rwc format is correctly
    converted to a dataframe."""
    rwc_df = wcqtlib.data.parse.rwc_to_dataframe(RWC_ROOT)
    yield __test_df_has_data, rwc_df
    yield __test_pd_output, rwc_df, RWC_ROOT, "rwc"


@pytest.mark.skipif(not os.path.exists(UIOWA_ROOT),
                    reason="Data not found.")
def test_uiowa_to_dataframe():
    """Test that an input folder with files in uiowa format is correctly
    converted to a dataframe."""
    uiowa_df = wcqtlib.data.parse.uiowa_to_dataframe(UIOWA_ROOT)
    yield __test_df_has_data, uiowa_df
    yield __test_pd_output, uiowa_df, UIOWA_ROOT, "uiowa"


@pytest.mark.skipif(not os.path.exists(PHIL_ROOT),
                    reason="Data not found.")
def test_philharmonia_to_dataframe():
    """Test that an input folder with files in philharmonia format is correctly
    converted to a dataframe."""
    philharmonia_df = wcqtlib.data.parse.philharmonia_to_dataframe(PHIL_ROOT)
    yield __test_df_has_data, philharmonia_df
    yield __test_pd_output, philharmonia_df, PHIL_ROOT, "philharmonia"


def test_normalize_instrument_names(classmap):
    example_data = {"instrument": classmap.allnames}
    df = pandas.DataFrame(example_data)

    norm_df = wcqtlib.data.parse.normalize_instrument_names(df)
    assert not norm_df.empty
    assert set(norm_df["instrument"].unique()) == set(classmap.classnames)


@pytest.mark.skipif(not all([os.path.exists(DATA_ROOT),
                             os.path.exists(PHIL_ROOT),
                             os.path.exists(UIOWA_ROOT),
                             os.path.exists(RWC_ROOT)]),
                    reason="Data not found.")
def test_load_dataframes():
    dfs = wcqtlib.data.parse.load_dataframes(DATA_ROOT)
    yield __test_df_has_data, dfs


@pytest.fixture
def classmap():
    return wcqtlib.data.parse.InstrumentClassMap()


def test_load_classmap(classmap):
    assert classmap is not None


def test_classmap_allnames(classmap):
    assert isinstance(classmap.allnames, list)
    assert len(classmap.allnames) > 1
    assert map(lambda x: isinstance(x, str), classmap.allnames)


def test_classmap_classnames(classmap):
    assert isinstance(classmap.classnames, list)
    assert len(classmap.classnames) == 12
    assert map(lambda x: isinstance(x, str), classmap.allnames)


def test_classmap_getattr(classmap):
    assert classmap["bassoon"] == "bassoon"
    assert classmap["acoustic-guitar"] == "guitar"
    assert classmap["Trumpet"] == "trumpet"


def test_classmap_index(classmap):
    assert classmap.get_index("bassoon") == 0
    assert classmap.from_index(0) == "bassoon"
    assert classmap.get_index("violin") == 11
    assert classmap.from_index(11) == "violin"
    assert classmap.size == 12


def test_classmap_indeces_match(classmap):
    for i in range(classmap.size):
        classname = classmap.from_index(i)
        assert classmap.get_index(classname) == i
