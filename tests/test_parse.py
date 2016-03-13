""" Test parse.py functions.

(cbj) I'm not really sure how to test these without the files existing, but
for starters, I'm going to do it by creating dummy hierarchies
in the same format.
"""

import os
import pandas
import pytest

import wcqtlib.data.parse


DATA_ROOT = os.path.expanduser("~/data")
RWC_ROOT = os.path.join(DATA_ROOT, "RWC Instruments")
UIOWA_ROOT = os.path.join(DATA_ROOT, "uiowa")
PHIL_ROOT = os.path.join(DATA_ROOT, "philharmonia")


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


def test_parse_phil_path():
    test_pairs = [("banjo_B3_very-long_piano_normal.mp3",
                   ("banjo", "B3", "very-long", "piano", "normal")),
                  ("cello_A3_1_fortissimo_arco-normal.mp3",
                   ("cello", "A3", "1", "fortissimo", "arco-normal")),
                  ("trumpet_A3_15_pianissimo_normal.mp3",
                   ("trumpet", "A3", "15", "pianissimo", "normal"))]

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
