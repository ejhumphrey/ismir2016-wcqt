import argparse
import logging
import pandas
import os
import sys

import wcqtlib.config as C
import wcqtlib.data.parse as parse
import wcqtlib.data.extract as E
import wcqtlib.data.cqt
import wcqtlib.train.driver as driver
import wcqtlib.common.utils as utils

CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                           "data", "master_config.yaml")

logger = logging.getLogger(__name__)


def collect(master_config):
    """Prepare dataframes of notes for experiments."""
    config = C.Config.from_yaml(master_config)

    print(utils.colored("Parsing directories to collect datasets"))
    parse_result = parse.parse_files_to_dataframe(config)

    print(utils.colored("Spliting audio files to notes."))
    extract_result = E.extract_notes(config)

    return all([parse_result,
                extract_result])


def extract_features(master_config):
    """Extract CQTs from all files collected in collect."""
    config = C.Config.from_yaml(master_config)
    print(utils.colored("Extracting CQTs from note audio."))
    success = wcqtlib.data.cqt.cqt_from_df(config, **config["features/cqt"])
    return success


def train(master_config,
          experiment_name):
    """Run training loop.

    Parameters
    ----------
    master_config : str
        Full path

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.
    """
    print(utils.colored("Training"))
    config = C.Config.from_yaml(master_config)

    model_definition = config["model"]
    hold_out_set = config["training/hold_out_set"]
    max_files_per_class = config.get(
        "training/max_files_per_class", None)

    driver.train_model(config,
                       model_selector=model_definition,
                       experiment_name=experiment_name,
                       hold_out_set=hold_out_set,
                       max_files_per_class=max_files_per_class)


def evaluate(master_config,
             experiment_name,
             select_epoch=None):
    """Evaluate datasets and report results.

    Parameters
    ----------
    master_config : str

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    select_epoch : str or None
        Which model params to select. Use the epoch number for this, for
        instance "1830" would use the model file "params1830.npz".
        If None, uses "final.npz"
    """
    print(utils.colored("Evaluating"))
    config = C.Config.from_yaml(master_config)

    selected_model_file = "params{}.npz".format(select_epoch) \
        if select_epoch else "final.npz"

    driver.evaluate_and_analyze(
        config, experiment_name, selected_model_file)


def analyze(master_config):
    """Analyze results from an experiment."""
    print(utils.colored("Analyzing"))
    raise NotImplementedError("analyze not yet implemented")


def notebook(master_config):
    """Launch the associated notebook."""
    print(utils.colored("Launching notebook."))
    raise NotImplementedError("notebook not yet implemented")


def test(master_config):
    """Launch all unit tests."""
    print(utils.colored("Running unit tests."))
    raise NotImplementedError("notebook not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master_config", default=CONFIG_PATH)

    subparsers = parser.add_subparsers()
    collect_parser = subparsers.add_parser('collect')
    collect_parser.set_defaults(func=collect)
    extract_features_parser = subparsers.add_parser('extract_features')
    extract_features_parser.set_defaults(func=extract_features)
    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('experiment_name',
                              help="Name of the experiment. "
                                   "Files go in a directory of this name.")
    train_parser.set_defaults(func=train)
    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.add_argument('experiment_name',
                                 help="Name of the experiment. "
                                      "Files go in a directory of this name.")
    evaluate_parser.add_argument('-s', '--select_epoch',
                                 default=None, type=int)
    evaluate_parser.set_defaults(func=evaluate)
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.set_defaults(func=analyze)
    notebook_parser = subparsers.add_parser('notebook')
    notebook_parser.set_defaults(func=notebook)
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)

    logging.basicConfig(level=logging.DEBUG)

    args = vars(parser.parse_args())
    fx = args.pop('func', None)
    if fx:
        success = fx(**args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

