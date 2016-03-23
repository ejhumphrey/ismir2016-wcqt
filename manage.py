import argparse
import logging
import logging.config
import numpy as np
import os
import pandas
import shutil
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


def run_process_if_not_exists(process, filepath, **kwargs):
    if not os.path.exists(filepath):
        return process(**kwargs)
    else:
        return True


def save_canonical_files(master_config):
    """Create the canonical_files.json file."""
    config = C.Config.from_yaml(master_config)
    datasets_path = os.path.join(
        os.path.expanduser(config['paths/extract_dir']),
        config['dataframes/datasets'])
    datasets_df = pandas.read_json(datasets_path)
    logger.info("Generating canonical datasets file from {} records".format(
        len(datasets_df)))
    success = parse.generate_canonical_files(datasets_df)
    logger.info("Success: {}".format(success))
    return success


def clean(master_config):
    """Clean dataframes and extracted audio/features."""
    config = C.Config.from_yaml(master_config)

    data_path = os.path.expanduser(config['paths/extract_dir'])
    # Clean data
    answer = input("Are you sure you want to delete {} (y|s to skip): ".format(data_path))
    if answer in ['y', 'Y']:
        shutil.rmtree(data_path)
        logger.info("clean done.")
    elif answer in ['s', 'S']:
        return True
    else:
        print("Exiting")
        sys.exit(1)


def collect(master_config, clean_data=True):
    """Prepare dataframes of notes for experiments."""
    config = C.Config.from_yaml(master_config)

    if clean_data:
        clean(master_config)

    print(utils.colored("Parsing directories to collect datasets"))
    parse_result = parse.parse_files_to_dataframe(config)
    return parse_result


def extract_notes(master_config):
    print(utils.colored("Spliting audio files to notes."))
    config = C.Config.from_yaml(master_config)

    skip_existing = config['extract/skip_existing']
    datasets_path = os.path.join(
        os.path.expanduser(config['paths/extract_dir']),
        config['dataframes/datasets'])
    run_process_if_not_exists(collect, datasets_path,
                              master_config=master_config,
                              clean_data=False)
    extract_result = E.extract_notes(config, skip_existing)

    return extract_result


def extract_features(master_config, skip_existing=True):
    """Extract CQTs from all files collected in collect."""
    config = C.Config.from_yaml(master_config)
    print(utils.colored("Extracting CQTs from note audio."))
    success = wcqtlib.data.cqt.cqt_from_df(config, **config["features/cqt"])
    return success


def fit_and_predict(master_config, experiment_name):
    """Runs:
    - train
    - model_selection_df
    - predict
    - analyze
    """
    # Step 1: train
    train(master_config, experiment_name)
    # Step 2: model selection
    results_df = model_selection(master_config, experiment_name)
    best_iter, best_param_file = driver.select_best_iteration(results_df)
    # Step 3: predictions
    predict(master_config, experiment_name, select_epoch=best_iter)
    # Step 4: analysis
    analyze(master_config, experiment_name, select_epoch=best_iter)


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
    hold_out_set = config["experiment/hold_out_set"]
    max_files_per_class = config.get(
        "training/max_files_per_class", None)

    driver.train_model(config,
                       model_selector=model_definition,
                       experiment_name=experiment_name,
                       hold_out_set=hold_out_set,
                       max_files_per_class=max_files_per_class)


def model_selection(master_config,
                    experiment_name,
                    plot_loss=False):
    """Perform model selection on the validation set.

    Parameters
    ----------
    master_config : str
        Full path

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    plot_loss : bool
        If true, uses matplotlib to non-blocking plot the loss
        at each validation.
    """
    print(utils.colored("Model Selection"))
    config = C.Config.from_yaml(master_config)

    hold_out_set = config["experiment/hold_out_set"]

    # load the valid_df of files to validate with.
    model_dir = os.path.join(
        os.path.expanduser(config["paths/model_dir"]),
        experiment_name)
    valid_df_path = os.path.join(
        model_dir, config['experiment/data_split_format'].format(
            "valid", hold_out_set))
    valid_df = pandas.read_pickle(valid_df_path)

    return driver.find_best_model(config,
                           experiment_name=experiment_name,
                           validation_df=valid_df,
                           plot_loss=plot_loss)


def predict(master_config,
            experiment_name,
            select_epoch=None):
    """Predict results on all datasets and report results.

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

    max_iterations = config['training/max_iterations']
    params_zero_pad = int(np.ceil(np.log10(max_iterations)))
    param_format_str = config['experiment/params_format']
    param_format_str = param_format_str.format(params_zero_pad)
    selected_model_file = param_format_str.format(select_epoch) \
        if str(select_epoch).isdigit() else "{}.npz".format(
            select_epoch)

    results = driver.predict(
        config, experiment_name, selected_model_file)
    logger.info("Generated results for {} files.".format(len(results)))


def analyze(master_config,
            experiment_name,
            select_epoch=None):
    """Predict results on all datasets and report results.

    Parameters
    ----------
    master_config : str

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    dataset : str
        Dataset to select results from for analysis.

    select_epoch : str or None
        Which model params to select. Use the epoch number for this, for
        instance "1830" would use the model file "params1830.npz".
        If None, uses "final.npz"
    """
    print(utils.colored("Analyzing"))
    config = C.Config.from_yaml(master_config)

    hold_out_set = config["experiment/hold_out_set"]

    driver.analyze(config, experiment_name, select_epoch, hold_out_set)
    return 0


def notebook(master_config):
    """Launch the associated notebook."""
    print(utils.colored("Launching notebook."))
    raise NotImplementedError("notebook not yet implemented")


def datatest(master_config, show_full=False):
    """Check your generated data."""
    config = C.Config.from_yaml(master_config)
    datasets_path = os.path.join(
        os.path.expanduser(config['paths/extract_dir']),
        config['dataframes/datasets'])

    # Regenerate the datasets_df to make sure it's not stale.
    if not collect(master_config, clean_data=False):
        logger.error("Failed to regenerate datasets_df.")

    datasets_df = pandas.read_json(datasets_path)
    logger.info("Your datasets_df has {} records".format(len(datasets_df)))

    canonical_df = parse.load_canonical_files()
    logger.info("Cannonical dataset has {} records".format(len(canonical_df)))

    diff_df = parse.diff_datasets_files(canonical_df, datasets_df)
    if len(diff_df) and show_full:
        print(utils.colored(
            "The following files are missing from your machine", "red"))

        print("Filename\tdirname\tdataset")
        for index, row in diff_df.iterrows():
            print("{:<40}\t{:<20}\t{:<15}".format(
                index, row['dirnamecan'], row['datasetcan']))
    else:
        print(utils.colored("You're all set; your dataset matches.", "green"))

    print(utils.colored("Now checking all files for validity."))

    classmap = wcqtlib.data.parse.InstrumentClassMap()
    filtered_df = datasets_df[datasets_df["instrument"].isin(
        classmap.allnames)]
    bad_files = E.check_valid_audio_files(filtered_df,
                                          write_path="bad_file_reads.txt")
    print(utils.colored("{} files could not be opened".format(
                        len(bad_files)), "red"))
    if bad_files:
        print("Unable to open the following audio files:")
        for filepath in bad_files:
            print(filepath)


def datastats(master_config):
    config = C.Config.from_yaml(master_config)
    canonical_df = parse.print_stats(config)


def test(master_config):
    """Launch all unit tests."""
    print(utils.colored("Running unit tests."))
    raise NotImplementedError("notebook not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master_config", default=CONFIG_PATH)

    subparsers = parser.add_subparsers()
    save_canonical_parser = subparsers.add_parser('save_canonical_files')
    save_canonical_parser.set_defaults(func=save_canonical_files)

    collect_parser = subparsers.add_parser('collect')
    collect_parser.set_defaults(func=collect)

    extract_notes_parser = subparsers.add_parser('extract_notes')
    extract_notes_parser.set_defaults(func=extract_notes)

    extract_features_parser = subparsers.add_parser('extract_features')
    extract_features_parser.set_defaults(func=extract_features)

    fit_and_predict_parser = subparsers.add_parser('fit_and_predict')
    fit_and_predict_parser.add_argument('experiment_name',
                                        help="Name of the experiment. "
                                        "Files go in a directory of "
                                        "this name.")
    fit_and_predict_parser.set_defaults(func=fit_and_predict)

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('experiment_name',
                              help="Name of the experiment. "
                                   "Files go in a directory of this name.")
    train_parser.set_defaults(func=train)
    modelselect_parser = subparsers.add_parser('model_selection')
    modelselect_parser.add_argument('experiment_name',
                                    help="Name of the experiment. "
                                    "Files go in a directory of this name.")
    modelselect_parser.add_argument('-p', '--plot_loss', action="store_true")
    modelselect_parser.set_defaults(func=model_selection)
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('experiment_name',
                                 help="Name of the experiment. "
                                      "Files go in a directory of this name.")
    predict_parser.add_argument('-s', '--select_epoch',
                                 default=None, type=int)
    predict_parser.set_defaults(func=predict)
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('experiment_name',
                                 help="Name of the experiment. "
                                      "Files go in a directory of this name.")
    analyze_parser.add_argument('-s', '--select_epoch',
                                 default=None)
    analyze_parser.set_defaults(func=analyze)
    notebook_parser = subparsers.add_parser('notebook')
    notebook_parser.set_defaults(func=notebook)

    # Tests
    datatest_parser = subparsers.add_parser('datatest')
    datatest_parser.add_argument('-p', '--show_full', action="store_true",
                                 help="Print the full diff to screen.")
    datatest_parser.set_defaults(func=datatest)
    datastats_parser = subparsers.add_parser('datastats')
    datastats_parser.set_defaults(func=datastats)
    test_parser = subparsers.add_parser('test')
    test_parser.set_defaults(func=test)

    # TODO MOve this to a file config.
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': "standard"
            },
        },
        'loggers': {
            '': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    })
    # logging.basicConfig(level=logging.DEBUG)

    args = vars(parser.parse_args())
    fx = args.pop('func', None)
    if fx:
        success = fx(**args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
