import numpy as np
import os
import pandas
import pytest

import wcqtlib.config as C
import wcqtlib.train.streams as streams

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path)


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_cqt_slices():
    def __test_valid_cqt(cqt_frame, expected_len):
        assert len(cqt_frame.shape) == 4
        assert cqt_frame.shape[1] == 1
        assert cqt_frame.shape[2] == expected_len

    t_len = 5
    # Sample the first avialble featuer file
    generator = streams.cqt_slices(features_df.iloc[0], t_len)
    data = next(generator)
    yield __test_valid_cqt, data['x_in'], t_len

    t_len = 1
    generator = streams.cqt_slices(features_df.iloc[1], t_len)
    data = next(generator)
    yield __test_valid_cqt, data['x_in'], t_len

    # Now for fun, do it for 100 of them just to make sure
    # it keeps working after one cycle of frames.
    t_len = 10
    generator = streams.cqt_slices(features_df.iloc[2], t_len)
    for i in range(50):
        data = next(generator)
        yield __test_valid_cqt, data['x_in'], t_len

    # Now do it for validate mode
    t_len = 8
    generator = streams.cqt_slices(features_df.iloc[3], t_len,
                                   shuffle=False, auto_restart=False)
    for frames in generator:
        yield __test_valid_cqt, frames['x_in'], t_len


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_wcqt_slices():
    def __test_valid_wcqt(wcqt_frame, expected_len):
        assert len(wcqt_frame.shape) == 4
        assert wcqt_frame.shape[0] == 1
        assert wcqt_frame.shape[2] == expected_len

    t_len = 5
    # Sample the first avialble featuer file
    generator = streams.wcqt_slices(features_df.iloc[0], t_len)
    data = next(generator)
    yield __test_valid_wcqt, data['x_in'], t_len

    t_len = 1
    generator = streams.wcqt_slices(features_df.iloc[1], t_len)
    data = next(generator)
    yield __test_valid_wcqt, data['x_in'], t_len

    # Now for fun, do it for 100 of them just to make sure
    # it keeps working after one cycle of frames.
    t_len = 10
    generator = streams.wcqt_slices(features_df.iloc[2], t_len,
                                    shuffle=False, auto_restart=False)
    for frame in generator:
        yield __test_valid_wcqt, frame['x_in'], t_len


@pytest.mark.xfail(reason="zmq is always generating batches of only one.")
def __test_streamer(streamer, t_len, batch_size):
    counter = 0
    while counter < 5:
        batch = next(streamer)
        assert batch is not None
        assert 'x_in' in batch and 'target' in batch
        assert len(batch['x_in'].shape) == 4
        assert len(batch['target'].shape) == 1
        assert batch['x_in'].shape[2] == t_len
        assert batch['x_in'].shape[0] == batch_size
        assert batch['target'].shape[0] == batch_size
        print("Result batch size:", batch['x_in'].shape[0])
        counter += 1


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer_cqt():
    t_len = 10
    batch_size = 12
    datasets = ["rwc"]
    streamer = streams.InstrumentStreamer(
        features_df, datasets, streams.cqt_slices,
        t_len=t_len, batch_size=batch_size)
    yield __test_streamer, streamer, t_len, batch_size


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer_wcqt():
    t_len = 10
    batch_size = 12
    datasets = ["rwc"]
    streamer = streams.InstrumentStreamer(
        features_df, datasets, streams.wcqt_slices,
        t_len=t_len, batch_size=batch_size)
    yield __test_streamer, streamer, t_len, batch_size


@pytest.mark.skipif(not all([os.path.exists(EXTRACT_ROOT),
                             os.path.exists(features_path),
                             not features_df.empty]),
                    reason="Data not found.")
def test_instrument_streamer_with_zmq():
    t_len = 10
    batch_size = 12
    datasets = ["rwc"]
    streamer = streams.InstrumentStreamer(
        features_df, datasets, streams.cqt_slices,
        t_len=t_len, batch_size=batch_size, use_zmq=True)
    yield __test_streamer, streamer, t_len, batch_size
