import datetime
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import shutil
from sklearn.cross_validation import train_test_split

import wcqtlib.config as C
import wcqtlib.common.utils as utils
import wcqtlib.analyze
import wcqtlib.train.models as models
import wcqtlib.train.streams as streams
import wcqtlib.train.evaluate as evaluate

logger = logging.getLogger(__name__)


class EarlyStoppingException(Exception):
    pass


def conditional_colored(value, minval, formatstr="{:0.3f}", color="green"):
    val_str = formatstr.format(float(value))
    if value < minval:
        val_str = utils.colored(val_str, color)
    return val_str


class TimerHolder(object):
    def __init__(self):
        self.timers = {}

    def start(self, tuple_or_list):
        """
        Note: tuples can be keys.
        Parameters
        ----------
        tuple_or_list : str or list of str
        """
        if isinstance(tuple_or_list, (str, tuple)):
            self.timers[tuple_or_list] = [datetime.datetime.now(), None]
        elif isinstance(tuple_or_list, list):
            for key in tuple_or_list:
                self.timers[key] = [datetime.datetime.now(), None]

    def end(self, tuple_or_list):
        """
        Parameters
        ----------
        tuple_or_list : str or list of str
        """
        if isinstance(tuple_or_list, (str, tuple)):
            self.timers[tuple_or_list][1] = datetime.datetime.now()
            return self.timers[tuple_or_list][1] - \
                self.timers[tuple_or_list][0]
        elif isinstance(tuple_or_list, list):
            results = []
            for key in tuple_or_list:
                self.timers[key][1] = datetime.datetime.now()
                results += [self.timers[key][1] - self.timers[key][0]]
            return results

    def get(self, key):
        if key in self.timers:
            if self.timers[key][1]:
                return self.timers[key][1] - self.timers[key][0]
            else:
                return self.timers[key][0]
        else:
            return None

    def get_start(self, key):
        return self.timers.get(key, None)[0]

    def get_end(self, key):
        return self.timers.get(key, None)[1]

    def mean(self, key_root, start_ind, end_ind):
        keys = [(key_root, x) for x in range(max(start_ind, 0), end_ind)]
        values = [self.get(k) for k in keys if k in self.timers]
        return np.mean(values)


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

    Trains for max_iterations or max_time, whichever is fewer.
    [Specified in the config.]

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
    logger.info("Starting training for experiment: {}".format(experiment_name))
    # Important paths & things to load.
    features_path = os.path.join(
        os.path.expanduser(config["paths/extract_dir"]),
        config["dataframes/features"])
    features_df = pandas.read_pickle(features_path)
    model_dir = os.path.join(
        os.path.expanduser(config["paths/model_dir"]),
        experiment_name)
    params_dir = os.path.join(model_dir, config["experiment/params_dir"])
    experiment_config_path = os.path.join(model_dir,
                                          config['experiment/config_path'])
    training_loss_path = os.path.join(model_dir,
                                      config['experiment/training_loss'])
    training_df_save_path = os.path.join(
        model_dir, config['experiment/data_split_format'].format(
            "train", hold_out_set))
    valid_df_save_path = os.path.join(
        model_dir, config['experiment/data_split_format'].format(
            "valid", hold_out_set))
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
    # Save the dfs to disk so we can use them for validation later.
    training_df.to_pickle(training_df_save_path)
    valid_df.to_pickle(valid_df_save_path)

    # Save the config we used in the model directory, just in case.
    config.save(experiment_config_path)

    # Duration parameters
    max_iterations = config['training/max_iterations']
    params_zero_pad = int(np.ceil(np.log10(max_iterations)))
    max_time = config['training/max_time']  # in seconds
    param_format_str = config['experiment/params_format']
    # insert the zero padding into the format string.
    param_format_str = param_format_str.format(params_zero_pad)

    # Collect various necessary parameters
    t_len = config['training/t_len']
    batch_size = config['training/batch_size']
    n_targets = config['training/n_targets']
    logger.debug("Hyperparams:\nt_len: {}\nbatch_size: {}\n"
                 "n_targets: {}\nmax_iterations: {}\nmax_time: {}s or {}h"
                 .format(t_len, batch_size, n_targets, max_iterations,
                         max_time, (max_time / 60. / 60.)))

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

    iter_print_freq = config.get('training/iteration_print_frequency', None)
    iter_write_freq = config.get('training/iteration_write_frequency', None)

    timers = TimerHolder()
    iter_count = 0
    train_stats = pandas.DataFrame(columns=['timestamp',
                                            'batch_train_dur',
                                            'iteration', 'loss'])
    min_train_loss = np.inf

    timers.start("train")
    logger.info("[{}] Beginning training loop at {}".format(
        experiment_name, timers.get("train")))
    try:
        timers.start(("stream", iter_count))
        for batch in streamer:
            timers.end(("stream", iter_count))
            timers.start(("batch_train", iter_count))
            loss = model.train(batch)
            timers.end(("batch_train", iter_count))
            row = dict(timestamp=timers.get_end(("batch_train", iter_count)),
                       batch_train_dur=timers.get(("batch_train", iter_count)),
                       iteration=iter_count,
                       loss=loss)
            train_stats.loc[len(train_stats)] = row

            # Time Logging
            logger.debug("[Iter timing] iter: {} | loss: {} | "
                         "stream: {} | train: {}".format(
                            iter_count, loss,
                            timers.get(("stream", iter_count)),
                            timers.get(("batch_train", iter_count))
                            ))
            # Print status
            if iter_print_freq and (iter_count % iter_print_freq == 0):
                mean_train_loss = train_stats["loss"][-iter_print_freq:].mean()
                logger.info("Iteration: {} | Mean_Train_loss: {}"
                            .format(iter_count,
                                    conditional_colored(mean_train_loss,
                                                        min_train_loss)))
                min_train_loss = min(mean_train_loss, min_train_loss)
                # Print the mean times for the last n frames
                logger.info("Mean stream time: {}, Mean train time: {}".format(
                    timers.mean("stream", iter_count - iter_print_freq,
                                iter_count),
                    timers.mean("batch_train", iter_count - iter_print_freq,
                                iter_count)))

            # save model, maybe
            if iter_write_freq and (iter_count % iter_write_freq == 0):
                save_path = os.path.join(
                    params_dir, param_format_str.format(iter_count))
                logger.debug("Writing params to {}".format(save_path))
                model.save(save_path)

            if datetime.datetime.now() > \
                    (timers.get("train") + datetime.timedelta(
                        seconds=max_time)):
                raise EarlyStoppingException("Max Time reached")

            iter_count += 1
            timers.start(("stream", iter_count))
            # Stopping conditions
            if (iter_count >= max_iterations):
                raise EarlyStoppingException("Max Iterations Reached")

    except KeyboardInterrupt:
        logger.warn(utils.colored("Training Cancelled", "red"))
        print("User cancelled training at epoch:", iter_count)
    except EarlyStoppingException as e:
        logger.warn(utils.colored("Training Stopped for {}".format(e), "red"))
        print("Training halted for: ", e)
    timers.end("train")

    # Print final training loss
    logger.info("Total iterations:".format(iter_count))
    logger.info("Trained for ".format(timers.get("train")))
    logger.info("Final training loss: {}".format(train_stats["loss"].iloc[-1]))

    # Make sure to save the final model.
    save_path = os.path.join(params_dir, "final.npz".format(iter_count))
    model.save(save_path)
    logger.info("Completed training for experiment: {}".format(
        experiment_name))

    # Save training loss
    logger.info("Writing training stats to {}".format(training_loss_path))
    train_stats.to_pickle(training_loss_path)


def find_best_model(config, experiment_name, validation_df, plot_loss=False):
    """Perform model selection on the validation set with a binary search
    for minimum validation loss.

    (Bayesean optimization might be another approach?)

    Parameters
    ----------
    master_config : str
        Full path

    experiment_name : str
        Name of the experiment. Files are saved in a folder of this name.

    validation_df : pandas.DataFrame
        Name of the held out dataset (used to specify the valid file)

    plot_loss : bool
        If true, uses matplotlib to non-blocking plot the loss
        at each validation.

    Returns
    -------
    results_df : pandas.DataFrame
        DataFrame containing the resulting losses.
    """
    logger.info("Finding best model for {}".format(
        utils.colored(experiment_name, "magenta")))
    # load the experiment config
    experiment_dir = os.path.join(
        os.path.expanduser(config['paths/model_dir']),
        experiment_name)
    model_dir = os.path.join(
        os.path.expanduser(config["paths/model_dir"]),
        experiment_name)
    params_dir = os.path.join(experiment_dir,
                              config['experiment/params_dir'])

    # load all necessary config parameters
    experiment_config_path = os.path.join(experiment_dir,
                                          config['experiment/config_path'])
    original_config = C.Config.from_yaml(experiment_config_path)
    slicer = get_slicer_from_network_def(original_config['model'])
    t_len = original_config['training/t_len']

    # Load the training loss
    training_loss_path = os.path.join(model_dir,
                                      config['experiment/training_loss'])
    training_loss = pandas.read_pickle(training_loss_path)
    validation_error_file = os.path.join(
        model_dir, original_config['experiment/validation_loss'])

    if not os.path.exists(validation_error_file):
        model_files = glob.glob(os.path.join(params_dir, "params*.npz"))
        # model_files.extend(glob.glob(os.path.join(params_dir, "final.npz")))

        result_df, best_model = evaluate.BinarySearchModelSelector(
            model_files, validation_df, slicer, t_len, show_progress=True)()

        result_df.to_pickle(validation_error_file)
        best_path = os.path.join(params_dir,
                                 original_config['experiment/best_params'])
        shutil.copyfile(best_model['model_file'], best_path)
    else:
        logger.info("Model Search already done; printing previous results")
        result_df = pandas.read_pickle(validation_error_file)
        #make sure model_iteration is an int so sorting makes sense.
        result_df["model_iteration"].apply(int)
        logger.info("\n{}".format(result_df.sort_values("model_iteration")))

    if plot_loss:
        fig = plt.figure()
        ax = plt.plot(training_loss["iteration"], training_loss["loss"])
        ax_val = plt.plot(result_df["model_iteration"], result_df["mean_loss"])
        plt.draw()
        plt.show()

    return result_df


def select_best_iteration(model_selection_df):
    """Given the model selection df, return the iteration
    which produced the best model.

    Returns
    -------
    best_model : int
        The iteration number which produced the best model.
    """
    best = model_selection_df.loc[model_selection_df["mean_acc"].argmax()]

    return best["model_iteration"], os.path.basename(best["model_file"])


def predict(config, experiment_name, selected_model_file,
            features_df_override=None):
    """Generates a prediction for *all* files, and writes them to disk
    as a dataframe.

    If features_df_override, replace the features_df with this
    dataframe (for testing)
    """
    logger.info("Evaluating experient {} with params from iter {}".format(
        utils.colored(experiment_name, "magenta"),
        utils.colored(selected_model_file, "cyan")))
    model_iter = utils.iter_from_params_filepath(selected_model_file)

    logger.info("Loading DataFrame...")
    if features_df_override is None:
        features_path = os.path.join(
            os.path.expanduser(config["paths/extract_dir"]),
            config["dataframes/features"])
        features_df = pandas.read_pickle(features_path)
    else:
        features_df = features_df_override

    experiment_dir = os.path.join(
        os.path.expanduser(config['paths/model_dir']),
        experiment_name)
    experiment_config_path = os.path.join(experiment_dir,
                                          config['experiment/config_path'])
    original_config = C.Config.from_yaml(experiment_config_path)
    params_file = os.path.join(experiment_dir,
                               config['experiment/params_dir'],
                               selected_model_file)
    slicer = get_slicer_from_network_def(original_config['model'])

    logger.info("Deserializing Network & Params...")
    model = models.NetworkManager.deserialize_npz(params_file)

    t_len = original_config['training/t_len']
    logger.info("Running evaluation on all files...")
    predictions_df = evaluate.evaluate_dataframe(features_df, model, slicer,
                                                 t_len, show_progress=True)
    predictions_df_path = os.path.join(
        experiment_dir,
        original_config.get('experiment/predictions_format',
                            config.get('experiment/predictions_format', None))
        .format(model_iter))
    predictions_df.to_pickle(predictions_df_path)
    return predictions_df


def analyze(config, experiment_name, model_name, hold_out_set):
    logger.info("Evaluating experient {} with params from {}".format(
        utils.colored(experiment_name, "magenta"),
        utils.colored(model_name, "cyan")))

    experiment_dir = os.path.join(
        os.path.expanduser(config['paths/model_dir']),
        experiment_name)
    experiment_config_path = os.path.join(experiment_dir,
                                          config['experiment/config_path'])
    original_config = C.Config.from_yaml(experiment_config_path)

    analyzer = wcqtlib.analyze.PredictionAnalyzer.from_config(
        original_config, experiment_name, model_name, hold_out_set)
    analysis_path = os.path.join(
        experiment_dir,
        original_config.get('experiment/analysis_format',
                            config.get('experiment/analysis_format', None))
        .format(model_name))
    logger.info("Saving analysis to:", analysis_path)
    analyzer.save(analysis_path)
    return analyzer
