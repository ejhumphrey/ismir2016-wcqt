import lasagne
import logging
import numpy as np
import os
import pandas
import pytest
import sys

import wcqtlib.config as C
import wcqtlib.train.streams as streams
import wcqtlib.train.models as models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path)


@pytest.fixture
def simple_network_def():
    input_shape = (None, 1, 5, 5)
    network_def = {
        "input_shape": input_shape,
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 4,
            "filter_size": (2, 2),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        },
        {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)
        },
        {
            "type": "layers.DropoutLayer",
            "p": 0.5
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 2,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def test_names_to_objects():
    test_dict = {
        "i'm_a_layer": "layers.Conv1DLayer",
        "i'm_a_list": [{
            "anotherlayer": "init.glorot"},
            {"yetanother": "i shouldn't work",
             "butishould": "nonlin.rectify"}
        ]
    }
    object_dict = models.names_to_objects(test_dict)
    assert object_dict["i'm_a_layer"] == lasagne.layers.Conv1DLayer
    assert isinstance(object_dict["i'm_a_list"][0]["anotherlayer"],
                      lasagne.init.GlorotUniform)
    assert object_dict["i'm_a_list"][1]["yetanother"] == \
        "i shouldn't work"
    assert object_dict["i'm_a_list"][1]["butishould"] == \
        lasagne.nonlinearities.rectify


def __test_network(network_def, input_shape):
    model = models.NetworkManager(network_def)
    assert model is not None

    batch_size = 8
    input_shape = (batch_size,) + input_shape[1:]
    test_batch = np.random.random(input_shape)
    test_target = np.asarray(np.random.randint(2, size=batch_size),
                             dtype=np.int32)
    loss = model.train_fx(test_batch, test_target)
    assert np.isfinite(loss)

    probs = model.predict_fx(test_batch)
    assert np.all(np.isfinite(probs))

    loss, acc = model.eval_fx(test_batch, test_target)
    assert np.isfinite(loss) and np.isfinite(acc)


def test_networkmanager_buildnetwork():
    # 1c 1f
    input_shape = (None, 1, 5, 5)
    network_def = {
        "input_shape": input_shape,
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 4,
            "filter_size": (2, 2),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        },
        {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)
        },
        {
            "type": "layers.DropoutLayer",
            "p": 0.5
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 2,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    yield __test_network, network_def, input_shape

    # 0c 3f
    input_shape = (None, 1, 8, 252)
    network_def = {
        "input_shape": input_shape,
        "layers": [
        {
            "type": "layers.DropoutLayer",
            "p": 0.25
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 512,
            "nonlinearity": "nonlin.rectify"
        },
        {
            "type": "layers.DropoutLayer",
            "p": 0.5
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 512,
            "nonlinearity": "nonlin.rectify"
        },
        {
            "type": "layers.DropoutLayer",
            "p": 0.5
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 2,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    yield __test_network, network_def, input_shape

    # 3c 3f
    {
        "input_shape": input_shape,
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 16,
            "filter_size": (1, 3),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"},
        {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (1, 2)  # 8 x 125
            },
        {
            "type": "layers.Conv2DLayer",
            "num_filters": 32,
            "filter_size": (3, 4),  # -> 6x122
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        },
        {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)  # -> 3 x 61
        },
        {
            "type": "layers.Conv2DLayer",
            "num_filters": 64,
            "filter_size": (2, 4),  # -> 2 x 58
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        },
        {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (1, 2)  # -> 2 x 29
        },
        {
            "type": "layers.DropoutLayer",
            "p": 0.5
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 256,
            "nonlinearity": "nonlin.rectify"
        },
        {
            "type": "layers.DropoutLayer",
            "p": 0.5
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 256,
            "nonlinearity": "nonlin.rectify"
        },
        {
            "type": "layers.DropoutLayer",
            "p": 0.5
        },
        {
            "type": "layers.DenseLayer",
            "num_units": 2,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    yield __test_network, network_def, input_shape


@pytest.mark.skip(reason="ToDo")
def test_networkmanager_updatehypers():
    pass


def test_networkmanager_save(workspace, simple_network_def):
    """Also test deserialization."""
    save_path = os.path.join(workspace, "output.npz")

    model = models.NetworkManager(simple_network_def)
    assert model is not None

    success = model.save(save_path)
    assert success and os.path.exists(save_path)

    data = np.load(save_path)
    assert "definition" in data and "params" in data

    model = models.NetworkManager.deserialize_npz(save_path)
    assert model is not None
    assert model._network is not None


def test_networkmanager_train_and_predict(simple_network_def):
    input_shape = (None, 1, 5, 5)

    model = models.NetworkManager(simple_network_def)
    assert model is not None

    batch_size = 8
    input_shape = (batch_size,) + input_shape[1:]

    batch = dict(
        x_in=np.random.random(input_shape),
        target=np.asarray(np.random.randint(2, size=batch_size),
                          dtype=np.int32))
    loss = model.train(batch)
    assert np.isfinite(loss)

    probs = model.predict(batch)
    assert np.all(np.isfinite(probs))

    loss, acc = model.evaluate(batch)
    assert np.isfinite(loss) and np.isfinite(acc)


@pytest.mark.cqt
@pytest.mark.slowtest
def test_overfit_two_samples_cqt():
    """Prove that our network works by training it with two random files
    from our dataset, intentionally overfitting it.

    Warning: not deterministic, but you could make it.
    """

    # Get list of instruments
    instruments = sorted(features_df["instrument"].unique())
    selected_instruments = instruments[:2]

    # Create a dataframe from our dataframe with only two files in it
    test_df = pandas.concat([
        features_df[features_df["instrument"] ==
                    selected_instruments[0]].sample(),
        features_df[features_df["instrument"] ==
                    selected_instruments[1]].sample()])

    t_len = 8
    batch_size = 8
    datasets = ["rwc"]
    n_targets = 2
    # Create a streamer that samples just those two files.
    streamer = streams.InstrumentStreamer(
        test_df, datasets, streams.cqt_slices,
        t_len=t_len, batch_size=batch_size)

    # Create a new model
    network_def = models.cqt_iX_c1f1_oY(t_len, n_targets)
    model = models.NetworkManager(network_def)

    # Train the model for N epochs, till it fits the damn thing
    max_batches = 250
    i = 0
    for batch in streamer:
        train_loss = model.train(batch)

        i += 1
        print("Batch: ", i, "Loss: ", train_loss)
        if i >= max_batches:
            break

    # Evaluate it. On the original files. Should do well.
    eval_batch = next(streamer)
    eval_probs = model.predict(eval_batch)
    eval_loss, accuracy = model.evaluate(eval_batch)
    print("Predictions:", eval_probs)
    print("Eval Loss:", eval_loss, "Accuracy:", accuracy)
    assert np.all(np.isfinite(eval_probs)) and np.isfinite(eval_loss) and \
        np.isfinite(accuracy)


@pytest.mark.wcqt
@pytest.mark.slowtest
def test_overfit_two_samples_wcqt():
    """Prove that our network works by training it with two random files
    from our dataset, intentionally overfitting it.

    Warning: not deterministic, but you could make it.
    """

    # Get list of instruments
    instruments = sorted(features_df["instrument"].unique())
    selected_instruments = instruments[:2]

    # Create a dataframe from our dataframe with only two files in it
    test_df = pandas.concat([
        features_df[features_df["instrument"] ==
                    selected_instruments[0]].sample(),
        features_df[features_df["instrument"] ==
                    selected_instruments[1]].sample()])

    t_len = 8
    batch_size = 8
    datasets = ["rwc"]
    n_targets = 2
    # Create a streamer that samples just those two files.
    streamer = streams.InstrumentStreamer(
        test_df, datasets, streams.wcqt_slices,
        t_len=t_len, batch_size=batch_size)
    input_shape = next(streamer)['x_in'].shape
    print(input_shape)

    # Create a new model
    network_def = models.wcqt_iX_c1f1_oY(t_len, n_targets)
    model = models.NetworkManager(network_def)

    # Train the model for N epochs, till it fits the damn thing
    max_batches = 250
    i = 0
    for batch in streamer:
        train_loss = model.train(batch)

        i += 1
        print("Batch: ", i, "Loss: ", train_loss)
        if i >= max_batches:
            break

    # Evaluate it. On the original files. Should do well.
    eval_batch = next(streamer)
    eval_probs = model.predict(eval_batch)
    eval_loss, accuracy = model.evaluate(eval_batch)
    print("Eval predictions:", eval_probs)
    print("Eval Loss:", eval_loss, "Accuracy:", accuracy)
    assert np.isfinite(eval_loss) and np.isfinite(accuracy)
