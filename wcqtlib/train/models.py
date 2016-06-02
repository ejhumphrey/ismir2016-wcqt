"""Model definitions. Each model definition takes an
    * n_in : int
    * n_out : int
  There are different models for cqt and wcqt ('cause you need to treat
  them differently).
  Each returns the training function, and the prediction function.
"""
import copy
import glob
import lasagne
import logging
import numpy as np
import os
import theano
import theano.tensor as T


logger = logging.getLogger(__name__)

CQT_DIMS = 204
WCQT_DIMS = (8, 36)
HCQT_DIMS = (6, 144)


__all__ = ['NetworkManager']

__layermap__ = {
    "layers.Conv1DLayer": lasagne.layers.Conv1DLayer,
    "layers.Conv2DLayer": lasagne.layers.Conv2DLayer,
    "layers.MaxPool2DLayer": lasagne.layers.MaxPool2DLayer,
    "layers.DenseLayer": lasagne.layers.DenseLayer,
    "layers.DropoutLayer": lasagne.layers.DropoutLayer,
    "nonlin.rectify": lasagne.nonlinearities.rectify,
    "nonlin.softmax": lasagne.nonlinearities.softmax,
    "init.glorot": lasagne.init.GlorotUniform(),
    "loss.categorical_crossentropy":
        lasagne.objectives.categorical_crossentropy
}


class InvalidNetworkDefinition(Exception):
    pass


class ParamLoadingError(Exception):
    pass


def list_experiments(config):
    """Given a master config, return a list of available models."""
    model_dir = os.path.expanduser(config['paths/model_dir'])
    logger.debug("Loading models from dir: {}".format(model_dir))
    experiments = [x for x in os.listdir(model_dir)
                   if os.path.isdir(os.path.join(model_dir, x))]
    logger.debug("Available Experiments: {}".format(experiments))
    experiment_contents = {}
    for experiment in experiments:
        loss_dfs = glob.glob(os.path.join(model_dir, experiment, "*_loss.pkl"))
        predictions = glob.glob(os.path.join(
                                model_dir, experiments, "*_predictions.pkl"))
        analysis = glob.glob(os.path.join(
                                model_dir, experiments, "*_analysis.pkl"))
        experiment_contents[experiments] = dict(
            loss_dfs=loss_dfs if loss_dfs else None,
            predictions=predictions if predictions else None,
            analysis=analysis if analysis else None)
    return experiment_contents


def names_to_objects(config_dict):
    """Given a configruation dict, convert all values which are strings
    in __layermap__.keys() to their value.
    """
    config_copy = copy.deepcopy(config_dict)
    for key, value in config_dict.items():
        if isinstance(value, str):
            # Replace it with the item in the map, or if it's not in the map
            #  just keep the value.
            config_copy[key] = __layermap__.get(value, value)
        elif isinstance(value, dict):
            # If it's a dict, call this recursively.
            config_copy[key] = names_to_objects(config_dict[key])
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    config_copy[key][i] = names_to_objects(config_dict[key][i])
    return config_copy


class NetworkManager(object):
    """Class for managing models, including:
     * creating them from a configuration and/or parameters
     * Serializing parameters & model conifiguration to disk
    """
    def __init__(self,
                 network_definition,
                 hyperparameters=None,
                 params=None):
        """
        Parameters
        ----------
        network_definition : dict or NetworkDefinition
        hyperparameters : dict or None
            If None, uses defaults.
        params : dict or None
            Serialized params to load in.
            If None, randomly initializes the model.
        """
        self.network_definition = network_definition
        self._init_hyperparam_defaults()
        if hyperparameters:
            self.update_hyperparameters(**hyperparameters)

        self._network = self._build_network()
        if params:
            self._load_params(params)

    def _init_hyperparam_defaults(self):
        self.hyperparams = {
            "learning_rate": theano.shared(lasagne.utils.floatX(0.01)),
            "momentum": theano.shared(lasagne.utils.floatX(.9))
        }

    @classmethod
    def deserialize_npz(cls, path):
        """
        Parameters
        ----------
        path : str
            Full path to a npz? containing the a network definition
            and optionally serialized params.
        """
        data = np.load(path)
        return cls(data['definition'].item(), params=data['params'].tolist())

    def _build_network(self):
        """Constructs the netork from the definition."""
        logger.debug("Building Netork")
        object_definition = names_to_objects(self.network_definition)
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        # Input layer is always just defined from the input shape.
        logger.debug("Building Netork - Input shape: {}".format(
            object_definition["input_shape"]))
        network = lasagne.layers.InputLayer(
            object_definition["input_shape"],
            input_var=input_var)

        # Now, construct the network from the definition.
        for layer in object_definition["layers"]:
            logger.debug("Building Netork - Layer: {}".format(
                object_definition["input_shape"]))
            layer_class = layer.pop("type", None)
            if not layer_class:
                raise InvalidNetworkDefinition(
                    "Each layer must contain a 'type'")

            if object_definition.get("batch_norm", False) is True \
                    and 'nonlinearity' in layer:
                network = lasagne.layers.batch_norm(
                    layer_class(network, **layer))
            else:
                network = layer_class(network, **layer)

        # save the network so we can intropect it in testing.
        self._network = network

        # Create loss functions for train and test.
        train_prediction = lasagne.layers.get_output(network)
        train_loss = object_definition['loss'](train_prediction, target_var)
        train_loss = train_loss.mean()

        # Collect params and update expressions.
        self.params = lasagne.layers.get_all_params(network, trainable=True)
        updates_rmsprop = lasagne.updates.rmsprop(
            train_loss, self.params,
            learning_rate=self.hyperparams["learning_rate"])
        updates_momentum = lasagne.updates.apply_nesterov_momentum(
            updates_rmsprop, self.params,
            momentum=self.hyperparams["momentum"])

        test_prediction = lasagne.layers.get_output(network,
                                                    deterministic=True)
        test_loss = object_definition['loss'](test_prediction, target_var)
        test_loss = train_loss.mean()
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        self.train_fx = theano.function(
            [input_var, target_var], train_loss, updates=updates_momentum)
        self.predict_fx = theano.function(
            [input_var], test_prediction)
        self.eval_fx = theano.function(
            [input_var, target_var], [test_loss, test_acc])

        return network

    def _load_params(self, params):
        """Loads the serialized parameters into model.
        The network must exist first, but there's no way you should be calling
        this unless that's true.
        """
        try:
            lasagne.layers.set_all_param_values(self._network, params)
        except ValueError:
            raise ParamLoadingError("Check to make sure your params "
                                    "match your model.")

    def update_hyperparameters(self, **hyperparams):
        """Update some hyperparameters."""
        for key, value in hyperparams.items():
            if key in self.hyperparams:
                self.hyperparams[key].set_value(value)

    def get_hyperparameter(self, name):
        return self.hyperparams[name].get_value()

    def save(self, write_path):
        """Write an npz containing the network definition and the parameters

        Parameters
        ----------
        write_path : str
            Full path to save to.

        Returns
        -------
        success : bool
        """
        param_values = lasagne.layers.get_all_param_values(self._network)
        np.savez(write_path,
                 definition=self.network_definition,
                 params=param_values)
        return os.path.exists(write_path)

    def train(self, batch):
        """Trains the network using a pescador batch.

        Parameters
        ----------
        batch : dict
            With at least keys:
            x_in : np.ndarray
            target : np.ndarray

            where len(x_in) == len(target)

        Returns
        -------
        training_loss : float
            The loss over this batch.
        """
        return self.train_fx(np.asarray(batch['x_in'],
                             dtype=theano.config.floatX),
                             np.asarray(batch['target'], dtype=np.int32))

    def predict(self, batch):
        """Predict values on a batch.

        Parameters
        ----------
        batch : dict
            With at least keys:
            x_in : np.ndarray

        Returns
        -------
        predictions : np.ndarray
            Returns the predictions for this batch.
        """
        return self.predict_fx(np.asarray(batch['x_in'],
                               dtype=theano.config.floatX))

    def evaluate(self, batch):
        """Get evaluation scores for a batch using the prediction.

        Parameters
        ----------
        batch : dict
            With at least keys:
            x_in : np.ndarray
            target : np.ndarray

            where len(x_in) == len(target)

        Returns
        -------
        prediction_loss : float
            The loss over this batch.
        prediction_acc : float
            The accuracy over this batch.
        """
        return self.eval_fx(np.asarray(batch['x_in'],
                            dtype=theano.config.floatX),
                            np.asarray(batch['target'], dtype=np.int32))


def cqt_iX_f1_oY(n_in, n_out):
    network_def = {
        "input_shape": (None, 1, n_in, CQT_DIMS),
        "layers": [{
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def cqt_iX_c1f1_oY(n_in, n_out):
    network_def = {
        "input_shape": (None, 1, n_in, CQT_DIMS),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 4,
            "filter_size": (1, 3),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 5)
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def cqt_iX_c2f2_oY(n_in, n_out):
    network_def = {
        "input_shape": (None, 1, n_in, CQT_DIMS),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 16,
            "filter_size": (3, 3),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)
        }, {
            "type": "layers.Conv2DLayer",
            "num_filters": 32,
            "filter_size": (1, 4),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (1, 2)
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": 128,
            "nonlinearity": "nonlin.rectify"
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def wcqt_iX_c1f1_oY(n_in, n_out):
    network_def = {
        "input_shape": (None, WCQT_DIMS[0], n_in, WCQT_DIMS[1]),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 4,
            "filter_size": (2, 3),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def wcqt_iX_c2f2_oY(n_in, n_out):
    network_def = {
        "input_shape": (None, WCQT_DIMS[0], n_in, WCQT_DIMS[1]),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 32,
            "filter_size": (3, 3),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)
        }, {
            "type": "layers.Conv2DLayer",
            "num_filters": 64,
            "filter_size": (1, 4),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (1, 2)
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": 256,
            "nonlinearity": "nonlin.rectify"
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def hcqt_iX_c1f1_oY(n_in, n_out):
    network_def = {
        "input_shape": (None, HCQT_DIMS[0], n_in, HCQT_DIMS[1]),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 4,
            "filter_size": (2, 3),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def hcqt_iX_c2f2_oY(n_in, n_out):
    network_def = {
        "input_shape": (None, HCQT_DIMS[0], n_in, HCQT_DIMS[1]),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": 32,
            "filter_size": (3, 3),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (2, 2)
        }, {
            "type": "layers.Conv2DLayer",
            "num_filters": 64,
            "filter_size": (1, 4),
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": (1, 2)
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": 256,
            "nonlinearity": "nonlin.rectify"
        }, {
            "type": "layers.DropoutLayer",
            "p": 0.5
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def
