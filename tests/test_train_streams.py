import librosa
import numpy as np
import os
import pandas as pd
import pytest

import hcnn.common.config as C
import hcnn.data.cqt
import hcnn.data.dataset
import hcnn.train.streams as streams

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.load(CONFIG_PATH)
RANDOM_SEED = 42


def __assert_cqt_slicer(dataset, t_len, *slicer_args):
    slicer = streams.cqt_slices(dataset.to_df().iloc[0], t_len,
                                *slicer_args, random_seed=RANDOM_SEED)

    for i in range(10):
        data = next(slicer)['x_in']
        assert len(data.shape) == 4
        assert data.shape[1] == 1
        assert data.shape[2] == t_len


def __assert_cqt_slicer_predict(dataset, t_len, *slicer_args):
    slicer = streams.cqt_slices(dataset.to_df().iloc[0], t_len,
                                *slicer_args, random_seed=RANDOM_SEED)

    # The first one should work
    data = next(slicer)['x_in']
    assert len(data.shape) == 4
    assert data.shape[1] == 1
    assert data.shape[2] == t_len

    # The second one should raise stopiteration
    with pytest.raises(StopIteration):
        data = next(slicer)['x_in']


def __assert_wcqt_slicer(dataset, t_len, *slicer_args):
    slicer = streams.wcqt_slices(dataset.to_df().iloc[0], t_len,
                                 *slicer_args, random_seed=RANDOM_SEED)
    for i in range(10):
        data = next(slicer)['x_in']
        assert len(data.shape) == 4
        assert data.shape[1] > 1 and data.shape[1] < 10
        assert data.shape[2] == t_len


def __assert_wcqt_slicer_predict(dataset, t_len, *slicer_args):
    slicer = streams.wcqt_slices(dataset.to_df().iloc[0], t_len,
                                 *slicer_args, random_seed=RANDOM_SEED)
    # The first one should work
    data = next(slicer)['x_in']
    assert len(data.shape) == 4
    assert data.shape[1] > 1 and data.shape[1] < 10
    assert data.shape[2] == t_len

    # The second one should raise stopiteration
    with pytest.raises(StopIteration):
        data = next(slicer)['x_in']


def __assert_hcqt_slicer(dataset, t_len, *slicer_args):
    slicer = streams.hcqt_slices(dataset.to_df().iloc[0], t_len,
                                 *slicer_args, random_seed=RANDOM_SEED)
    for i in range(10):
        data = next(slicer)['x_in']
        assert len(data.shape) == 4
        assert data.shape[1] == 3
        assert data.shape[2] == t_len


def __assert_hcqt_slicer_predict(dataset, t_len, *slicer_args):
    slicer = streams.hcqt_slices(dataset.to_df().iloc[0], t_len,
                                 *slicer_args, random_seed=RANDOM_SEED)
    # The first one should work
    data = next(slicer)['x_in']
    assert len(data.shape) == 4
    assert data.shape[1] == 3
    assert data.shape[2] == t_len

    # The second one should raise stopiteration
    with pytest.raises(StopIteration):
        data = next(slicer)['x_in']


@pytest.mark.parametrize(
    "slicer_test",
    [__assert_cqt_slicer,
     __assert_wcqt_slicer,
     __assert_hcqt_slicer],
    ids=["cqt", "wcqt", "hcqt"])
def test_slices_train_mode(slicer_test, tiny_feats):
    slicer_test(tiny_feats, 5)
    slicer_test(tiny_feats, 1)
    slicer_test(tiny_feats, 10)
    slicer_test(tiny_feats, 43)


@pytest.mark.parametrize(
    "slicer_test",
    [__assert_cqt_slicer_predict,
     __assert_wcqt_slicer_predict,
     __assert_hcqt_slicer_predict],
    ids=["cqt", "wcqt", "hcqt"])
def test_slices_predict_mode(slicer_test, tiny_feats):
    slicer_test(tiny_feats, 5, False, False, False)
    slicer_test(tiny_feats, 1, False, False, False)
    slicer_test(tiny_feats, 10, False, False, False)
    slicer_test(tiny_feats, 43, False, False, False)
    slicer_test(tiny_feats, 8, False, False, False)


@pytest.fixture(params=[30, 43, 50],
                ids=["30=too_small", "43=just_right", "50=too_big"])
def generated_data(request, workspace):
    t_len = 43
    cqt_size = hcnn.data.cqt.CQT_PARAMS['n_bins']

    # make some garbage data
    vector_size = request.param
    test_vector = np.random.random((1, vector_size, cqt_size))
    testfile = os.path.join(workspace, "foo.npz")
    np.savez(testfile, cqt=test_vector)
    # Also make an example dataframe
    df = pd.DataFrame({'cqt': testfile, 'instrument': 'trumpet'},
                      index=[0], columns=['cqt', 'instrument'])
    return df, t_len


def test_cqt_slicer_with_data_less_tlen(generated_data):
    df, t_len = generated_data
    slicer = streams.cqt_slices(df.iloc[0], t_len)
    batch = next(slicer)
    assert batch['x_in'].shape[2] == t_len


def test_wcqt_slicer_with_data_less_tlen(generated_data):
    df, t_len = generated_data
    slicer = streams.wcqt_slices(df.iloc[0], t_len)
    batch = next(slicer)
    assert batch['x_in'].shape[2] == t_len


def test_cqt_slicer_add_noise_has_appropriate_scale(tiny_feats):
    record = tiny_feats.to_df().iloc[0]
    original_cqt = np.load(record['cqt'])['cqt']
    original_len = original_cqt.shape[-2]
    # mimic the cqt in the slicer.
    original_logcqt = librosa.logamplitude(original_cqt ** 2, ref_power=np.max)
    original_mean = original_logcqt.mean()

    t_len = original_len + 10
    slicer = streams.cqt_slices(record, t_len)
    batch = next(slicer)
    assert batch['x_in'].shape[2] == t_len
    new_mean = batch['x_in'][:, :, original_len:, :].mean()
    assert new_mean < original_mean


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


@pytest.mark.parametrize("slicer", [streams.cqt_slices,
                                    streams.wcqt_slices,
                                    streams.hcqt_slices],
                         ids=["cqt", "wcqt", "hcqt"])
def test_instrument_streamer(slicer, tiny_feats):
    df = tiny_feats.to_df()
    t_len = 10
    batch_size = 12
    if not df.empty:
        streamer = streams.InstrumentStreamer(
            df, slicer, t_len=t_len, batch_size=batch_size)
        __test_streamer(streamer, t_len, batch_size)


@pytest.mark.xfail(reason="zmq is always generating batches of only one.")
def test_instrument_streamer_with_zmq(tiny_feats):
    df = tiny_feats.to_df()
    t_len = 10
    batch_size = 12
    if not df.empty:
        streamer = streams.InstrumentStreamer(
            df, streams.cqt_slices,
            t_len=t_len, batch_size=batch_size, use_zmq=True)
        __test_streamer(streamer, t_len, batch_size)
