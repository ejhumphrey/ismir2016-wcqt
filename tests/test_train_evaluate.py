import copy
import logging
import numpy as np
import os
import pandas
import pytest
from sklearn.metrics import classification_report
import sys

import wcqtlib.config as C
import wcqtlib.train.evaluate as evaluate
import wcqtlib.train.models as models
import wcqtlib.train.streams as streams

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), os.pardir,
                           "data", "master_config.yaml")
config = C.Config.from_yaml(CONFIG_PATH)

EXTRACT_ROOT = os.path.expanduser(config['paths/extract_dir'])
features_path = os.path.join(EXTRACT_ROOT, config['dataframes/features'])
features_df = pandas.read_pickle(features_path)


@pytest.fixture(params=[(streams.cqt_slices, models.cqt_iX_c1f1_oY),
                        (streams.wcqt_slices, models.wcqt_iX_c1f1_oY)],
                ids=["cqt", "wcqt"])
def slicer_and_model(request):
    return request.param


def test_evaluate_one(slicer_and_model):
    # TODO: random seeds for consistency / reproducability.
    print("Running on:")
    # Get a record
    test_record = features_df.iloc[42]
    other_record = features_df.iloc[142]
    assert test_record['instrument'] != other_record['instrument']

    # Pick a model
    t_len = 8
    n_classes = 12
    slicer = slicer_and_model[0]
    network_def = slicer_and_model[1](t_len, n_classes)
    model = models.NetworkManager(network_def)

    # Train this a couple frames to bias the eval in favor of this file.
    iter_count = 0
    max_count = 10
    for frames in slicer(test_record, t_len):
        loss = model.train(frames)
        print("Quick train ", iter_count, "loss:", loss)
        iter_count += 1
        if iter_count >= max_count:
            break

    result = evaluate.evaluate_one(test_record, model, slicer, t_len)
    for key in ["mean_loss", "mean_acc", "max_likelyhood", "vote", "target"]:
        assert key in result
    assert result['max_likelyhood'] == result['target']
    assert result['vote'] == result['target']
    print("Test Result:\n", result)

    other = evaluate.evaluate_one(other_record, model, slicer, t_len)
    for key in ["mean_loss", "mean_acc", "max_likelyhood", "vote", "target"]:
        assert key in other
    print("Other Result:\n", other)


def test_evalute_dataframe(slicer_and_model):
    # For the purposes of this we don't care too much about what we train with.
    # TODO: random seeds for consistency / reproducability.
    test_df = features_df.sample(n=(12*8))

    # Pick a model
    t_len = 8
    n_classes = 12
    slicer = slicer_and_model[0]
    network_def = slicer_and_model[1](t_len, n_classes)
    model = models.NetworkManager(network_def)

    # Create the streamer.
    streamer = streams.InstrumentStreamer(test_df, ["rwc", "uiowa"],
                                          record_slicer=slicer,
                                          t_len=t_len,
                                          batch_size=12)

    # Train for a little bit.
    iter_count = 0
    max_count = 100
    for batch in streamer:
        loss = model.train(batch)
        print("Batch ", iter_count, "loss:", loss)
        iter_count += 1
        if iter_count >= max_count:
            break

    # Run evaluation on some number of datapoints (10ish),
    #  and make sure you get a dataframe back
    eval_df = evaluate.evaluate_dataframe(test_df, model, slicer, t_len)
    assert isinstance(eval_df, pandas.DataFrame)
    assert len(eval_df) == len(test_df)

    print("File Class Predictions", np.bincount(eval_df["max_likelyhood"]))
    print("File Class Targets", np.bincount(eval_df["target"]))
    print(classification_report(eval_df["max_likelyhood"].tolist(),
                                eval_df["target"].tolist()))

    analysis = evaluate.analyze_results(eval_df, "unit test")
    assert not analysis.empty
    print(analysis)
