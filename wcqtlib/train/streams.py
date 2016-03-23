"""
Generators/Streams for generating data.
"""

import collections
import logging
import numpy as np
import pescador

import wcqtlib.data.extract as extract
import wcqtlib.data.parse as parse
import wcqtlib.common.utils as utils

logger = logging.getLogger(__name__)

instrument_map = parse.InstrumentClassMap()


def cqt_slices(record, t_len, shuffle=True, auto_restart=True):
    """Generate slices from a cqt file.

    Thresholds the frames by finding the cqt means, and
    only returning frames with means greater than the lowest
    quarter of the means.

    To use this for training, use the following parameters:
        cqt_slices(..., shuffle=True, auto_restart=True)
    To use this for prediction / eval, use the following parameters:
        cqt_slices(..., shuffle=False, auto_restart=False)

    Parameters
    ----------
    record : pandas.Series
        Single pandas record containing a 'cqt' record
        which points to the cqt file in question.
        Also must contain an "instrument" column
        which contains the ground truth.

    t_len : int
        Length of the sliced array [in time/frames]

    shuffle : bool
        If True, shuffles the frames every time through the file.
        If False, Reads frames from start fo finish.

    auto_restart : bool
        If True, yields infinitely.
        If False, only goes through the file once.

    Yields
    -------
    sample : dict with fields {x_in, label}
        The windowed observation.
    """
    # Load the npz
    cqt = np.load(record['cqt'])['cqt']
    target = instrument_map.get_index(record["instrument"])

    num_obs = cqt.shape[1] - t_len
    # If there aren't enough obs, don't do it.
    if num_obs > 0:
        # Get the frame means, and remove the lowest 1/4
        cqt_mu = cqt[0, :num_obs].mean(axis=1)
        threshold = sorted(cqt_mu)[int(len(cqt_mu)*.25)]
        idx = (cqt_mu[:num_obs] >= threshold).nonzero()[0]
        if shuffle:
            np.random.shuffle(idx)

        counter = 0
        while True:
            obs = utils.slice_ndarray(cqt, idx[counter], length=t_len, axis=1)
            data = dict(
                x_in=obs[np.newaxis, ...],
                target=np.asarray((target,)))
            yield data

            # Once we have used all of the frames once, reshuffle.
            counter += 1
            if counter >= len(idx):
                if not auto_restart:
                    break
                if shuffle:
                    np.random.shuffle(idx)
                counter = 0
    else:
        logger.warning("File {} doesn't have enough obs for t_len {}".format(
            record['cqt'], t_len))


def wcqt_slices(record, t_len, shuffle=True, auto_restart=True,
                p_len=36, p_stride=24):
    """Generate slices of wrapped CQT observations.

    To use this for training, use the following parameters:
        wcqt_slices(..., shuffle=True, auto_restart=True, ...)
    To use this for prediction / eval, use the following parameters:
        wcqt_slices(..., shuffle=False, auto_restart=False, ...)

    Parameters
    ----------
    stash : biggie.Stash
        Stash to draw the datapoint from.

    t_len : int
        Length of the CQT slice in time.

    shuffle : bool
        If True, shuffles the frames every time through the file.
        If False, Reads frames from start fo finish.

    auto_restart : bool
        If True, yields infinitely.
        If False, only goes through the file once.

    p_len : int
        Number of adjacent pitch bins.

    p_stride : int
        Number of pitch bins to stride when wrapping.

    Yields
    ------
    sample : dict with fields {x_in, label}
        The windowed observation.
    """
    # Load the npz
    cqt = np.load(record['cqt'])['cqt']
    wcqt = utils.fold_array(cqt[0], length=p_len, stride=p_stride)
    target = instrument_map.get_index(record["instrument"])

    num_obs = wcqt.shape[1] - t_len

    # If there aren't enough obs, don't do it.
    if num_obs > 0:
        # Get the frame means, and remove the lowest 1/4
        cqt_mu = cqt[0, :num_obs].mean(axis=1)
        threshold = sorted(cqt_mu)[int(len(cqt_mu)*.25)]
        idx = (cqt_mu[:num_obs] >= threshold).nonzero()[0]
        if shuffle:
            np.random.shuffle(idx)

        counter = 0
        while True:
            obs = utils.slice_ndarray(wcqt, idx[counter], length=t_len, axis=1)
            data = dict(
                x_in=obs[np.newaxis, ...],
                target=np.asarray((target,), dtype=np.int32))
            yield data

            # Once we have used all of the frames once, reshuffle.
            counter += 1
            if counter >= len(idx):
                if not auto_restart:
                    break
                if shuffle:
                    np.random.shuffle(idx)
                counter = 0
    else:
        logger.warning("File {} doesn't have enough obs for t_len {}".format(
            record['cqt'], t_len))


def buffer_stream(stream, batch_size):
    """Buffer stream into ndarrays.

    Parameters
    ----------
    stream : pescador.Stream

    batch_size : int
        Number of elements to return.

    Yields
    ------
    batch : dict
        Dense ndarrays.
    """
    return pescador.buffer_streamer(stream, batch_size)


def zmq_buffered_stream(stream, batch_size):
    """Buffer stream using zmq in a separate python process.

    Parameters
    ----------
    stream : pescador.Stream

    batch_size : int
        Number of elements to return.

    Yields
    ------
    batch : dict
        Dense ndarrays
    """
    return pescador.zmq_stream(stream, max_batches=batch_size)


class InstrumentStreamer(collections.Iterator):
    """Class wrapping the creation of a pescador streamer
    to sample equally from each instrument class available in
    the features dataframe.

    This class will behave like a generator."""

    def __init__(self, features_df, datasets,
                 record_slicer,
                 slicer_kwargs={},
                 t_len=1,
                 instrument_mux_params=dict(k=10, lam=20),
                 master_mux_params=dict(
                    n_samples=None,  # no maximum number
                    k=12,  # this is the number of classes.
                    lam=None,  # only use these 12 streams.
                    with_replacement=False,  # really, only use those 12
                    revive=True  # make sure only one copy of each is active
                                 # at a time.
                    ),
                 batch_size=50,
                 use_zmq=False):
        """
        Parameters
        ----------
        features_df : pandas.DataFrame
            Dataframe which points to the features files and has ground
            truth data.

        datasets : list of str
            Dataset(s) to use in the features_df.

        record_slicer : function
            Function used to load slices from a df record.
            (See {cqt_slices, wcqt} slices above).

        slicer_kwargs : dict
            Any parameters to send to the slider.
            t_len will always get fowarded to the slicer.

        t_len : int
            Number of frames in the time dimension to return for each sample.

        instrument_mux_params : dict
            Arguments to pass to the pescador.mux's which sample from
            each instrument class.

        master_mux_params : dict
            Arguments to pass to the pescador.mux's which sample from
            all of the instrument mux streams.

        batch_size : int
            Size of the batch to return from the buffered generator stream.

        use_zmq : bool
            If true, use a zmq_stream as the final sampler, in order
            to generate the data in parallel.
        """
        self.features_df = features_df
        self.datasets = datasets
        self.record_slicer = record_slicer
        self.slicer_kwargs = slicer_kwargs
        self.t_len = t_len
        self.instrument_mux_params = instrument_mux_params
        self.master_mux_params = master_mux_params
        self.batch_size = batch_size
        self.use_zmq = use_zmq

        self.setup()

    def setup(self):
        """Perform the setup to prepare for streaming."""
        # Instrument names
        instruments = list(self.features_df["instrument"].unique())

        # Get Muxes for each instrument.
        inst_muxes = [self._instrument_mux(i) for i in instruments]

        # Construct the streams for each mux.
        mux_streams = [pescador.Streamer(x) for x in inst_muxes \
                       if x is not None]

        # Construct the master mux
        master_mux = pescador.mux(mux_streams, **self.master_mux_params)
        # We have to wrap the mux in a stream so that the buffer
        #  knows what to do with it.
        self.master_stream = pescador.Streamer(master_mux)

        # Now construct the final streamer
        if self.use_zmq:
            self.buffered_streamer = zmq_buffered_stream(
                self.master_stream, self.batch_size)
        else:
            self.buffered_streamer = buffer_stream(
                self.master_stream, self.batch_size)

    def _instrument_streams(self, instrument):
        """Return a list of generators for all records in the dataframe
        which match the instrument given, and are in the datasets.

        Parameters
        ----------
        instrument : str
            Instrument to select.

        Returns
        -------
        streams : list of pescador.Streamer
            One streamer for each instrument file.
        """
        # Get list of instruments
        instrument_records = extract.filter_df(
            self.features_df, instrument=instrument, datasets=self.datasets)
        seed_pool = [pescador.Streamer(self.record_slicer, record, self.t_len,
                                       **self.slicer_kwargs)
                     for _, record in instrument_records.iterrows()]
        return seed_pool

    def _instrument_mux(self, instrument):
        """Return a pescador.mux for a single instrument.

        Parameters
        ----------
        instrument : str
            Instrument to select.

        Returns
        -------
        mux : pescador.mux
            A pescador.mux for a single instrument.
        """
        streams = self._instrument_streams(instrument)
        if len(streams):
            return pescador.mux(streams, n_samples=None,
                                **self.instrument_mux_params)
        else:
            return None

    def __iter__(self):
        return self.buffered_streamer

    def __next__(self):
        """Generate batches of samples and return them."""
        return next(self.buffered_streamer)
