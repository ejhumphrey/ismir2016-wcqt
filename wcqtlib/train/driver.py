import logging
import numpy as np
import os
import pandas

import wcqtlib.common.utils as utils
import wcqtlib.train.models as models
import wcqtlib.train.streams as streams

logger = logging.getLogger(__name__)


def construct_training_df(features_df, datasets, max_files_per_class):
    if max_files_per_class:
        search_df = features_df[features_df["dataset"].isin(datasets)]
        selected_instruments = []
        for instrument in search_df["instrument"].unique():
            selected_instruments.append(
                search_df[search_df["instrument"] == instrument].sample(
                    n=max_files_per_class))
        return pandas.concat(selected_instruments)
    else:
        return features_df


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
    training_df = construct_training_df(features_df,
                                        datasets,
                                        max_files_per_class)
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

    if 'wcqt' in model_selector:
        slicer = streams.wcqt_slices
    else:
        slicer = streams.wcqt_slices

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

    logger.info("[{}] Beginning training loop".format(experiment_name))
    epoch_count = 0
    epoch_mean_loss = []
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
            # save model, maybe
            if param_write_freq and (epoch_count % param_write_freq == 0):
                save_path = os.path.join(
                    params_dir, "params{0:0>4}.npz".format(epoch_count))
                model.save(save_path)

            epoch_count += 1
    except KeyboardInterrupt:
        print("User cancelled training at epoch:", epoch_count)

    # Print final training & validation loss & acc
    print("Final training loss:", epoch_mean_loss[-1])
    # Make sure to save the final model.
    save_path = os.path.join(params_dir, "final.npz".format(epoch_count))
    model.save(save_path)
    logger.info("Completed training for experiment:", experiment_name)
