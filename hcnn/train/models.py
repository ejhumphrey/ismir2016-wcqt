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
from lasagne.regularization import regularize_layer_params
import logging
import numpy as np
import os
import theano
import theano.tensor as T

from ..data import cqt


logger = logging.getLogger(__name__)

BINS_PER_OCTAVE = cqt.CQT_PARAMS['bins_per_octave']
CQT_DIMS = cqt.CQT_PARAMS['n_bins']
WCQT_DIMS = (CQT_DIMS // BINS_PER_OCTAVE, (BINS_PER_OCTAVE * 2) // 3)
HCQT_DIMS = (cqt.HARMONIC_PARAMS['n_harmonics'], CQT_DIMS)


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
        lasagne.objectives.categorical_crossentropy,
    "l1": lasagne.regularization.l1,
    "l2": lasagne.regularization.l2
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
            layer_batchnorm_override = layer.pop('batch_norm', True)

            if not layer_class:
                raise InvalidNetworkDefinition(
                    "Each layer must contain a 'type'")

            # This looks at the global batch_norm setting
            if (object_definition.get("batch_norm", False) is True and
                    # This checks the local batch_norm setting;
                    # allows override in the last layer.
                    layer_batchnorm_override is True):
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

        if object_definition.get('regularize', None):
            penalty = regularize_layer_params(network,
                                              object_definition['regularize'])
            train_loss = train_loss + object_definition.get('l2_weight', 1e-4) * penalty

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
        "layers": [
        {
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
            "type": "layers.DenseLayer",
            "num_units": 128,
            "nonlinearity": "nonlin.rectify"
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
            "type":  "layers.MaxPool2DLayer",
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
            "type": "layers.DenseLayer",
            "num_units": 256,
            "nonlinearity": "nonlin.rectify"
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
            "type": "layers.DenseLayer",
            "num_units": 256,
            "nonlinearity": "nonlin.rectify"
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax"
        }],
        "loss": "loss.categorical_crossentropy"
    }
    return network_def


def cqt_MF(n_in, n_out, n_basefilt):
    """Model definition for "MF"=> Full-height cqt filters.
    """
    L1_FILTER = (7, CQT_DIMS)
    L2_FILTER = (9, 1)
    network_def = {
        # (n=n_in, 252)
        # All parameter Math is for base case of *16 filters
        # (different than the others)
        # For 1s, 512 samples @ 22050, there are ~43 windows / sample.
        # All comment calculations assume 43 for n_in
        "input_shape": (None, 1, n_in, CQT_DIMS),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": n_basefilt,
            "filter_size": L1_FILTER,
            # => (n_basefilt, 37, 1)
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
            # parameters: (7 * 252 * 1) * 16 => 28,224
        }, {
            "type": "layers.Conv2DLayer",
            "num_filters": n_basefilt * 2,
            "filter_size": L2_FILTER,
            # => (8, 29, 1)
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
            # parameters: (9 * 1 * 16) * 32 => 4,608
        }, {
            "type": "layers.MaxPool2DLayer",
            # Pool: (29, 1)
            "pool_size": ((n_in - L1_FILTER[0] - L2_FILTER[0] + 2), 1)
            # => (32, 1, 1) = 32
        }, {
            "type": "layers.DenseLayer",
            "num_units": 1024,
            "nonlinearity": "nonlin.rectify"
            # 32 * 1024 => 32,768
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax",
            # 1024 * 12 [n_out] = 12,288
            "batch_norm": False
        }],
        "loss": "loss.categorical_crossentropy",
        "batch_norm": True,
        "regularize": "l2"
        # Total = 28,224 + 4,608 + 32,768 + 12,288 = 77,888 total params.
    }
    return network_def


def cqt_MF_n16(n_in, n_out):
    return cqt_MF(n_in, n_out, 16)


def cqt_MF_n32(n_in, n_out):
    return cqt_MF(n_in, n_out, 32)


def cqt_MF_n64(n_in, n_out):
    return cqt_MF(n_in, n_out, 64)


def cqt_M2(n_in, n_out, n_basefilt):
    """Model definition for M2: small 2-d CQT filters
    """
    L1_FILTER = (7, 7 * HCQT_DIMS[0])
    L2_FILTER = (9, 9)
    network_def = {
        # (n=n_in, 252)
        # All parameter Math is for base case of 8 filters
        # For 1s, 512 samples @ 22050, there are ~43 windows / sample.
        # All comment calculations assume 43 for n_in
        "input_shape": (None, 1, n_in, CQT_DIMS),
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": n_basefilt,
            "filter_size": L1_FILTER,
            # => (n_basefilt, 37, 232)
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
            # parameters: (7 * 21 * 1) * 8 => 1,176
        }, {
            "type": "layers.Conv2DLayer",
            "num_filters": n_basefilt * 2,
            "filter_size": L2_FILTER,
            # => (8, 29, 224)
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
            # parameters: (9 * 9 * 16) * 8 => 10,368
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": ((n_in - L1_FILTER[0] - L2_FILTER[0] + 2), 7)
            # => (16, 1, 32) = 512
        }, {
            "type": "layers.DenseLayer",
            "num_units": 128,
            "nonlinearity": "nonlin.rectify"
            # 512 * 128 => 65,536
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax",
            "batch_norm": False
            # 128 * 12 [n_out] = 1,536
        }],
        "loss": "loss.categorical_crossentropy",
        "batch_norm": True,
        "regularize": "l2"
        # Total = 1,176 + 10,368 + 65,536 + 1,536 = 78,616 total params.
    }
    return network_def


def cqt_M2_n8(n_in, n_out):
    return cqt_M2(n_in, n_out, 8)


def cqt_M2_n16(n_in, n_out):
    return cqt_M2(n_in, n_out, 16)


def cqt_M2_n32(n_in, n_out):
    return cqt_M2(n_in, n_out, 32)


def cqt_M2_n64(n_in, n_out):
    return cqt_M2(n_in, n_out, 64)


def hcqt_MH(n_in, n_out, n_basefilt):
    """Model definition for MH: small 3-d HCQT filters
    """
    L1_FILTER = (7, 7)
    L2_FILTER = (9, 9)
    network_def = {
        # (n=n_in, 252)
        # All parameter Math is for base case of 8 filters
        # For 1s, 512 samples @ 22050, there are ~43 windows / sample.
        # All comment calculations assume 43 for n_in
        "input_shape": (None, HCQT_DIMS[0], n_in, HCQT_DIMS[1]),
        #  => (3, 43, 252)
        "layers": [{
            "type": "layers.Conv2DLayer",
            "num_filters": n_basefilt,
            "filter_size": L1_FILTER,
            # => (n_basefilt, 37, 246)
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
            # parameters: (7 * 7 * 3) * 8 => 1,176
        }, {
            "type": "layers.Conv2DLayer",
            "num_filters": n_basefilt * 2,
            "filter_size": L2_FILTER,
            # => (8, 29, 238)
            "nonlinearity": "nonlin.rectify",
            "W": "init.glorot"
            # parameters: (9 * 9 * 16) * 8 => 10,368
        }, {
            "type": "layers.MaxPool2DLayer",
            "pool_size": ((n_in - L1_FILTER[0] - L2_FILTER[0] + 2), 7)
            # => (16, 1, 34) = 544
        }, {
            "type": "layers.DenseLayer",
            # Adjusted to 120 from 128 to account for more parameters
            # earlier, to try and match the # parameters more or less
            # to the M2 model.
            "num_units": 120,
            "nonlinearity": "nonlin.rectify"
            # 544 * 120 => 65,280
        }, {
            "type": "layers.DenseLayer",
            "num_units": n_out,
            "nonlinearity": "nonlin.softmax",
            "batch_norm": False
            # 128 * 12 [n_out] = 1,536
        }],
        "loss": "loss.categorical_crossentropy",
        "batch_norm": True,
        "regularize": "l2"
        # Total = 1,176 + 10,368 + 65,280 + 1,536 = 78360 total params.
    }
    return network_def


def hcqt_MH_n8(n_in, n_out):
    return hcqt_MH(n_in, n_out, 8)


def hcqt_MH_n16(n_in, n_out):
    return hcqt_MH(n_in, n_out, 16)


def hcqt_MH_n32(n_in, n_out):
    return hcqt_MH(n_in, n_out, 32)


def hcqt_MH_n64(n_in, n_out):
    return hcqt_MH(n_in, n_out, 64)
