"""
Generators/Streams for generating data.
"""

import collections
import librosa
import logging
import numpy as np
import os
import pescador
import zipfile

import hcnn.common.utils as utils
import hcnn.common.labels as labels

logger = logging.getLogger(__name__)

instrument_map = labels.InstrumentClassMap()


def base_slicer(record, t_len, obs_slicer, shuffle=True, auto_restart=True,
                add_noise=True, random_seed=None, npz_data_key='cqt',
                slice_logger=None):
    """Base slicer function for yielding data from an .npz file
    which contains the features for streaming.

    Assumptions:
     * Input file is about 1s in length (43 frames), but might be
       shorter or longer. t_len should be 43 (~1s), however.
     * It is possible that this file could get sampled more than
       once consecutively, so for training, if shuffle is True,
       and the number of frames in the file > t_len, it will
       try to sample from frames other than the first t_len frames
       randomly.
     * For prediction (shuffle=False), it will always return
       the first t_len frames.

    To use this for training, use the following parameters:
        (...shuffle=True, auto_restart=True, add_noise=True)
    For prediction / eval, use the following parameters:
        (...shuffle=False, auto_restart=False, add_noise=False)

    Parameters
    ----------
    record : pandas.Series
        Single pandas record containing a 'cqt' record
        which points to the cqt file in question.
        Also must contain an "instrument" column
        which contains the ground truth.

    t_len : int
        Length of the sliced array [in time/frames]

    obs_slicer : function
        Function which takes (cqt, idx, counter, t_len), and
        returns a slice from the data, formatted correctly for
        the desired data.

    shuffle : bool
        If True, shuffles the frames every time through the file.
        If False, returns the first t_len frames.

    auto_restart : bool
        If True, yields infinitely.
        If False, only goes through the file once.

    add_noise : bool
        If True, adds a small amount of noise to every sample.
        If False, does nothing.

    random_seed : int or None
        If int, uses this number as the random seed. Otherwise,
        makes it's own.

    npz_data_key : str
        The key in the npz file pointed to by record from which to
        load the data. Choices = ['cqt', 'harmonic_cqt']

    Yields
    -------
    sample : dict with fields {x_in, target}
        The windowed observation.
    """
    rng = np.random.RandomState(random_seed)
    target = instrument_map.get_index(record['instrument'])
    if slice_logger: slice_logger.start(record, target)

    if not (('cqt' in record.index) and
            record['cqt'] is not None and
            isinstance(record['cqt'], str) and
            os.path.exists(record['cqt'])):
        logger.error('No valid feature file specified for record: {}'.format(
            record))
        if slice_logger: slice_logger.error(record)
        return

    # Load the npz file with they key specified.
    try:
        cqt = np.load(record['cqt'])[npz_data_key]
    except zipfile.BadZipFile:
        logger.error('This zip file is bad! Sadness :(  -  {}'.format(
            record['cqt']))
        if slice_logger: slice_logger.error(record)
        return

    # Take the logmagnitude of the cqt
    cqt = librosa.logamplitude(cqt ** 2, ref_power=np.max)

    # Make sure the data is long enough.
    # In practice this should no longer be necessary.
    cqt = utils.backfill_noise(cqt, t_len + 1, mu=-80)

    num_possible_obs = cqt.shape[-2] - t_len
    if shuffle:
        idx = np.arange(num_possible_obs)
        rng.shuffle(idx)
    else:
        idx = np.arange(1)

    counter = 0
    while True:
        if slice_logger: slice_logger.sample(record)

        obs = obs_slicer(cqt, idx, counter, t_len)
        if add_noise:
            obs = obs + utils.same_shape_noise(obs, 1, rng)
        data = dict(
            x_in=obs,
            target=np.atleast_1d((target,)))
        yield data

        counter += 1
        if counter >= len(idx):
            if not auto_restart:
                break
            if shuffle:
                rng.shuffle(idx)
            counter = 0

    if slice_logger: slice_logger.close(record)


def cqt_slices(record, t_len, shuffle=True, auto_restart=True,
               add_noise=True, random_seed=None,
               slice_logger=None):
    """Generate slices from a cqt file.

    Thresholds the frames by finding the cqt means, and
    only returning frames with means greater than the lowest
    quarter of the means.

    To use this for training, use the following parameters:
        cqt_slices(..., shuffle=True, auto_restart=True, add_noise=True)
    To use this for prediction / eval, use the following parameters:
        cqt_slices(..., shuffle=False, auto_restart=False, , add_noise=False)

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

    random_seed : int
        If true, uses this number as the random seed. Otherwise,
        makes it's own.

    add_noise : bool

    Yields
    -------
    sample : dict with fields {x_in, target}
        The windowed observation.
    """
    def cqt_slicer(cqt, idx, counter, t_len):
        obs = utils.slice_ndarray(cqt, idx[counter], length=t_len, axis=1)
        return obs[np.newaxis, ...]

    for cqt_slice in base_slicer(
            record, t_len, cqt_slicer,
            shuffle=shuffle, auto_restart=auto_restart,
            add_noise=add_noise, random_seed=random_seed,
            npz_data_key='cqt',
            slice_logger=slice_logger):
        yield cqt_slice


def wcqt_slices(record, t_len, shuffle=True, auto_restart=True, add_noise=True,
                p_len=54, p_stride=36, random_seed=None, slice_logger=None):
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

    add_noise : bool

    p_len : int
        Number of adjacent pitch bins.

    p_stride : int
        Number of pitch bins to stride when wrapping.


    random_seed : int or None
         Override the random seed if given.

    Yields
    ------
    sample : dict with fields {x_in, label}
        The windowed observation.
    """
    def wcqt_slicer(cqt, idx, counter, t_len):
        # Grab the obs
        obs = utils.slice_ndarray(cqt, idx[counter], length=t_len, axis=1)
        # Convert it to WCQT
        wcqt = utils.fold_array(obs[0], length=p_len, stride=p_stride)
        # Fix the shape.s
        return wcqt[np.newaxis, ...]

    for wcqt_slice in base_slicer(
            record, t_len, wcqt_slicer,
            shuffle=shuffle, auto_restart=auto_restart,
            add_noise=add_noise, random_seed=random_seed,
            npz_data_key='cqt',
            slice_logger=slice_logger):
        yield wcqt_slice


def hcqt_slices(record, t_len, shuffle=True, auto_restart=True, add_noise=True,
                random_seed=None, slice_logger=None):
    """Generate slices of pre-generated harmonic cqts from a cqt file.

    To use this for training, use the following parameters:
        hcqt_slices(..., shuffle=True, auto_restart=True, add_noise=True)
    To use this for prediction / eval, use the following parameters:
        hcqt_slices(..., shuffle=False, auto_restart=False, add_noise=False)

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

    add_noise : bool

    random_seed : int or None
         Override the random seed if given.

    Yields
    -------
    sample : dict with fields {x_in, label}
        The windowed observation.
    """
    def hcqt_slicer(cqt, idx, counter, t_len):
        # hcqt = np.swapaxes(cqt, 1, 2)
        # Grab the obs
        obs = utils.slice_ndarray(cqt, idx[counter], length=t_len, axis=-2)
        # Fix the shape.
        return obs[np.newaxis, ...] if np.ndim(obs) == 3 else obs

    for hcqt_slice in base_slicer(
            record, t_len, hcqt_slicer,
            shuffle=shuffle, auto_restart=auto_restart,
            add_noise=add_noise, random_seed=random_seed,
            npz_data_key='harmonic_cqt',
            slice_logger=slice_logger):
        yield hcqt_slice


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

    def __init__(self, features_df,
                 record_slicer,
                 slicer_kwargs={},
                 t_len=1,
                 # 15 streams open, ~3 samples per stream before opening
                 #  a new one.
                 instrument_mux_params=dict(k=100, lam=1),
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
        mux_streams = [pescador.Streamer(x) for x in inst_muxes
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
        instrument_records = utils.filter_df(
            self.features_df, instrument=instrument)
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

    def next(self):
        return self.__next__()
