import logging
import numpy as np
import os
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

import wcqtlib.config as C
import wcqtlib.common.utils as utils
import wcqtlib.train.models as models
import wcqtlib.train.streams as streams
import wcqtlib.train.evaluate as evaluate

logger = logging.getLogger(__name__)


def construct_training_valid_df(features_df, datasets,
                                validation_split=0.2,
                                max_files_per_class=None):
    """Prepare training and validation dataframes from the features df.
     * First selects from only the datasets given in datasets.
     * Then **for each instrument** (so the distribution from each instrument
        doesn't change)
        * train_test_split to generate training and validation sets.
        * if max_files_per_class, also then restrict it to
            a maximum of that number of files for each train and test

    Parameters
    ----------
    features_df : pandas.DataFrame

    datasets : list of str
        List of datasets to use.

    validation_spilt : float
        Train/validation split.

    max_files_per_class : int
        Number of files to restrict the dataset to.

    """
    search_df = features_df[features_df["dataset"].isin(datasets)]

    selected_instruments_train = []
    selected_instruments_valid = []
    for instrument in search_df["instrument"].unique():
        instrument_df = search_df[search_df["instrument"] == instrument]
        traindf, validdf = train_test_split(
            instrument_df, test_size=validation_split)

        if max_files_per_class:
            traindf = traindf.sample(n=max_files_per_class)
            validdf = traindf.sample(n=max_files_per_class)

        selected_instruments_train.append(traindf)
        selected_instruments_valid.append(validdf)

    return pandas.concat(selected_instruments_train), \
        pandas.concat(selected_instruments_valid)


def get_slicer_from_network_def(network_def_name):
    if 'wcqt' in network_def_name:
        slicer = streams.wcqt_slices
    else:
        slicer = streams.cqt_slices
    return slicer


def train_model(config, model_selector, experiment_name,
                hold_out_set,
                max_files_per_class=None):
    """
    Train a model, writing intermediate params
    to disk.

    Trains for max_epochs epochs, where an epoch is:
     # 44 is the approximate number of average frames in one file.
     total_dataset_frames = n_training_files * (44 / t_len)
     epoch_size = total_dataset_frames / batch_size

    config: wcqtlib.config.Config
        Instantiated config.

    model_selector : str
        Name of the model to use.
        (This is a function name from models.py)

    experiment_name : str
        Name of the experiment. This is used to
        name the files/parameters saved.

    hold_out_set : str or list of str
        Which dataset to leave out in training.

    max_files_per_class : int or None
        Used for overfitting the network during testing;
        limit the training set to this number of files
        per class.
    """
    logger.info("Starting training for experiment:", experiment_name)
    # Important paths & things to load.
    features_path = os.path.join(
        os.path.expanduser(config["paths/extract_dir"]),
        config["dataframes/features"])
    features_df = pandas.read_pickle(features_path)
    model_dir = os.path.join(
        os.path.expanduser(config["paths/model_dir"]),
        experiment_name)
    params_dir = os.path.join(model_dir, "params")
    utils.create_directory(model_dir)
    utils.create_directory(params_dir)

    # Get the datasets to use excluding the holdout set.
    exclude_set = set(hold_out_set)
    datasets = set(features_df["dataset"].unique())
    datasets = datasets - exclude_set

    # Set up the dataframe we're going to train with.
    logger.info("[{}] Constructing training df".format(experiment_name))
    training_df, valid_df = construct_training_valid_df(
        features_df, datasets, max_files_per_class=max_files_per_class)
    logger.debug("[{}] training_df : {} rows".format(experiment_name,
                                                     len(training_df)))

    # Save the config we used in the model directory, just in case.
    config.save(os.path.join(model_dir, "config.yaml"))

    # Collect various necessary parameters
    t_len = config['training/t_len']
    batch_size = config['training/batch_size']
    n_targets = config['training/n_targets']
    max_epochs = config['training/max_epochs']
    epoch_length = int(len(training_df) * (44 / float(t_len)) /
                       float(batch_size))
    logger.debug("Hyperparams:\nt_len: {}\nbatch_size: {}\n"
                 "n_targets: {}\nmax_epochs: {}\nepoch_length: {}"
                 .format(t_len, batch_size, n_targets, max_epochs,
                         epoch_length))

    slicer = get_slicer_from_network_def(model_selector)

    # Set up our streamer
    logger.info("[{}] Setting up streamer".format(experiment_name))
    streamer = streams.InstrumentStreamer(
        training_df, datasets, slicer, t_len=t_len,
        batch_size=batch_size)

    # create our model
    logger.info("[{}] Setting up model: {}".format(experiment_name,
                                                   model_selector))
    network_def = getattr(models, model_selector)(t_len, n_targets)
    model = models.NetworkManager(network_def)

    batch_print_freq = config.get('training/train_print_frequency_batches',
                                  None)
    param_write_freq = config.get('training/param_write_frequency_epochs',
                                  None)
    predict_freq = config.get('training/predict_frequency_epochs',
                              None)

    logger.info("[{}] Beginning training loop".format(experiment_name))
    epoch_count = 0
    epoch_mean_loss = []
    validation_losses = []
    try:
        while epoch_count < max_epochs:
            logger.debug("Beginning epoch: ", epoch_count)
            # train, storing loss for each batchself.
            batch_count = 0
            epoch_losses = []
            for batch in streamer:
                logger.debug("Beginning ")
                train_loss = model.train(batch)
                epoch_losses += [train_loss]

                if batch_print_freq and (batch_count % batch_print_freq == 0):
                    print("Epoch: {} | Batch: {} | Train_loss: {}"
                          .format(epoch_count, batch_count, train_loss))

                batch_count += 1
                if batch_count >= epoch_length:
                    break
            epoch_mean_loss += [np.mean(epoch_losses)]

            # print valid, maybe
            if predict_freq and (epoch_count % predict_freq == 0):
                eval_df = evaluate.evaluate_dataframe(
                    valid_df, model, slicer, t_len)
                val_loss = eval_df['mean_loss'].mean()
                val_acc = eval_df['mean_acc'].mean()
                validation_losses += [val_loss]
                print("Epoch {} | Train Loss: [{:0.3f}] | Validation Loss "
                      "[{:0.3f}] | Acc [{:0.3f}]".format(
                        epoch_count, epoch_mean_loss[-1], val_loss, val_acc))

            # save model, maybe
            if param_write_freq and (epoch_count % param_write_freq == 0):
                save_path = os.path.join(
                    params_dir, "params{0:0>4}.npz".format(epoch_count))
                model.save(save_path)

            epoch_count += 1
    except KeyboardInterrupt:
        print("User cancelled training at epoch:", epoch_count)

    # Print final training & validation loss & acc
    print("Last Epoch:", epoch_count)
    print("Final training loss:", epoch_mean_loss[-1])
    print("Final validation loss:", validation_losses[-1])
    # Make sure to save the final model.
    save_path = os.path.join(params_dir, "final.npz".format(epoch_count))
    model.save(save_path)
    logger.info("Completed training for experiment:", experiment_name)


def evaluate_and_analyze(config, experiment_name, selected_model_file):
    print("Evaluating experient {} with params from {}".format(
        utils.colored(experiment_name, "magenta"),
        utils.colored(selected_model_file, "cyan")))

    print("Loading DataFrame...")
    features_path = os.path.join(
        os.path.expanduser(config["paths/extract_dir"]),
        config["dataframes/features"])
    features_df = pandas.read_pickle(features_path)

    experiment_dir = os.path.join(
        os.path.expanduser(config['paths/model_dir']),
        experiment_name)
    experiment_config_path = os.path.join(experiment_dir, "config.yaml")
    original_config = C.Config.from_yaml(experiment_config_path)
    params_file = os.path.join(experiment_dir, "params", selected_model_file)
    slicer = get_slicer_from_network_def(original_config['model'])

    print("Deserializing Network & Params...")
    model = models.NetworkManager.deserialize_npz(params_file)

    t_len = original_config['training/t_len']
    print("Running evaluation on all files...")
    eval_df = evaluate.evaluate_dataframe(features_df, model, slicer, t_len,
                                          show_progress=True)
    print("Calculating results...")
    results = evaluate.analyze_results(eval_df, experiment_name)
    print("{:*^30}".format(utils.colored("Results", "green")))
    print(results)

    print("File Class Predictions", np.bincount(eval_df["max_likelyhood"]))
    print("File Class Targets", np.bincount(eval_df["target"]))
    print(classification_report(eval_df["max_likelyhood"].tolist(),
                                eval_df["target"].tolist()))

    print("Random baseline should be: {:0.3f}".format(
          1.0 / original_config['training/n_targets']))
