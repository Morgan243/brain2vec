import os
import attr

import pandas as pd
import numpy as np
import scipy.io
import torch
import torchvision.transforms
from torch.utils import data as tdata

from tqdm.auto import tqdm

from typing import List, Optional, Type, ClassVar

from mmz import utils
from sklearn.pipeline import Pipeline
from brain2vec.preprocessing import steps as ps
from brain2vec import preprocessing as proc

from brain2vec.datasets import BaseDataset

with_logger = utils.with_logger(prefix_name=__name__)

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')

os.environ['WANDB_CONSOLE'] = 'off'

logger = utils.get_logger(__name__)


@attr.s
@with_logger
class BaseASPEN(BaseDataset):
    env_key = None
    default_base_path = None
    all_patient_maps = None
    default_sensor_columns = list(range(64))
    default_audio_sample_rate = 48000
    default_signal_sample_rate = 1200

    mat_d_keys = dict(
        signal=None,
        signal_fs=None,
        audio='audio',
        audio_fs='fs_audio',
        stimcode='stimcode',
        electrodes='electrodes',
        wordcode='wordcode')

    patient_tuples = attr.ib(None)
    sensor_columns = attr.ib(None)
    base_path = attr.ib(None)
    data_subset = attr.ib('Data')

    num_mfcc = attr.ib(13)

    selected_flat_indices = attr.ib(None)
    transform = attr.ib(None)
    transform_l = attr.ib(attr.Factory(list))
    all_input_transform = attr.ib(None)
    all_input_transform_l = attr.ib(attr.Factory(list))

    target_transform = attr.ib(None)
    target_transform_l = attr.ib(attr.Factory(list))

    flatten_sensors_to_samples = attr.ib(False)
    extra_output_keys = attr.ib(None)
    post_processing_func = attr.ib(None)
    post_processing_kws = attr.ib(attr.Factory(dict))

    # power_threshold = attr.ib(0.007)
    # power_q = attr.ib(0.70)
    pre_processing_pipeline = attr.ib(None)
    # If using one source of data, with different `selected_word_indices`, then
    # passing the first NWW dataset to all subsequent ones built on the same source data
    # can save on memory and reading+parsing time
    data_from: 'BaseASPEN' = attr.ib(None)

    label_reindex_col: str = attr.ib(None)#"patient"
    label_reindex_map: Optional[dict] = attr.ib(None)

    initialize_data = attr.ib(True)
    selected_flat_keys = attr.ib(None, init=False)
    label_reindex_ix: Optional[int] = attr.ib(None, init=False)

    default_data_subset = 'Data'
    default_location = None
    default_patient = None
    default_session = None
    default_trial = None

    def __attrs_post_init__(self):
        self.logger.debug(f"preparing pipeline")
        # Build pipelines based on this NWW dataset state
        self.pipeline_map = self.make_pipeline_map()
        self.logger.debug(f"Available pipelines: {list(self.pipeline_map.keys())}")

        if self.initialize_data:
            self.initialize()

    def initialize(self):

        # If nothing passed, use 'default' pipeline
        if self.pre_processing_pipeline is None:
            self.logger.info("Default pipeline selected")
            self.pipeline_f = self.pipeline_map['default']
        # If string passed, use it to select the pipeline in the map
        elif isinstance(self.pre_processing_pipeline, str):
            self.logger.info(f"'{self.pre_processing_pipeline}' pipeline selected")
            self.pipeline_f = self.pipeline_map[self.pre_processing_pipeline]
        # Otherwise, just assume it will work, that a callable is passed
        # TODO: Check that pipeline_f is callable
        else:
            self.logger.info(f"{str(self.pre_processing_pipeline)} pipeline passed directly")
            self.pipeline_f = self.pre_processing_pipeline

        # If the pipeline function is an sklearn pipeline, then use its transform() method
        if isinstance(self.pipeline_f, Pipeline):
            self.pipeline_obj = self.pipeline_f
            self.pipeline_f = self.pipeline_obj.transform

        # If no data sharing, then load and parse data from scratch
        if self.data_from is None:
            self.logger.info("Loading data directly")
            # Leave this here for now...
            # self.mfcc_m = torchaudio.transforms.MFCC(self.default_audio_sample_rate,
            #                                         self.num_mfcc)

            # - Data Loading -
            # - Load the data in its raw form from files - default is matlab files, override to change
            data_iter = tqdm(self.patient_tuples, desc="Loading data")
            mat_data_maps = {l_p_s_t_tuple: self.load_data(*l_p_s_t_tuple,
                                                           # sensor_columns=self.sensor_columns,
                                                           subset=self.data_subset)
                             for l_p_s_t_tuple in data_iter}

            # - Important processing - #
            # - Process each subject in data map through pipeline func
            # Init mapping from <patient tuple> -> pipeline output
            self.data_maps = dict()
            # Init mapping from <patient tuple> -> to pipelines 'sample_index_map' output (target->TimedeltaIndex)
            self.sample_index_maps = dict()
            for k, dmap in mat_data_maps.items():
                # Run the pipeline, mutating/modifying the data map for this patient trial
                res_dmap = self.pipeline_f(dmap)
                self.sample_index_maps[k] = res_dmap['sample_index_map']
                # The first data map sets the sampling frequency fs if it's not already set
                self.fs_signal = res_dmap[self.mat_d_keys['signal_fs']] if self.fs_signal is None else self.fs_signal
                # The first data map sets the N-samples per window if it's not already set
                self.n_samples_per_window = getattr(self, 'n_samples_per_window', res_dmap['n_samples_per_window'])
                self.logger.info(f"N samples per window: {self.n_samples_per_window}")

                # Check that signal frequencies always agree
                if self.fs_signal != res_dmap[self.mat_d_keys['signal_fs']]:
                    raise ValueError("Mismatch fs (%s!=%s) on %s" % (self.fs_signal, res_dmap['fs_signal'], str(k)))

                self.data_maps[k] = res_dmap

            # ----
            # SENSOR SELECTION LOGIC - based on the patients loaded - which sensors do we use?
            if self.sensor_columns is None or isinstance(self.sensor_columns, str):
                # Get each participant's good and bad sensor columns into a dictionary
                good_and_bad_tuple_d = {l_p_s_t_tuple: (mat_d['good_sensor_columns'], mat_d['bad_sensor_columns'])
                                        for l_p_s_t_tuple, mat_d in self.data_maps.items()}

                # Go back through and any missing sets of good_sensors are replaced with all sensor from the data
                good_and_bad_tuple_d = {
                        # Take good sensors (gs) if they were explicitly identified by the pipeline
                    k: (set(gs) if gs is not None
                        # Otherwise, assume all sensors are good
                        else (list(range(self.data_maps[k][self.mat_d_keys['signal']].shape[1]))),
                        # Second element of the tuple will still be the bad sensors (bs), which may be None
                        bs)
                    for k, (gs, bs) in good_and_bad_tuple_d.items()}

                self.logger.info("GOOD AND BAD SENSORS: " + str(good_and_bad_tuple_d))
                # Default to 'union' options if sensor columns not explicitly provided
                self.sensor_columns = 'union' if self.sensor_columns is None else self.sensor_columns

                # UNION: Select all good sensors from all inputs, zeros will be filled for those missing
                if self.sensor_columns == 'union':
                    # Create a sorted list of all sensor IDs found in the good sensor sets extracted
                    # Only allow good sensors that are not in the bad sensor list
                    self.selected_columns = sorted(list({_gs for k, (gs, bs) in good_and_bad_tuple_d.items()
                                                         for _gs in gs if bs is None or _gs not in bs}))
                # INTERSECTION: Select only sensors that are rated good in all inputs
                elif self.sensor_columns == 'intersection' or self.sensor_columns == 'valid':
                    s = [set(gs) for k, (gs, bs) in good_and_bad_tuple_d.items()]
                    self.selected_columns = sorted(list(s[0].intersection(*s[1:])))
                else:
                    raise ValueError("Unknown sensor columns argument: " + str(self.sensor_columns))
                self.logger.info(f"Selected {len(self.selected_columns)} columns using {self.sensor_columns} method: "
                                 f"{', '.join(map(str, self.selected_columns))}")
            else:
                self.selected_columns = self.sensor_columns

            self.sensor_count = len(self.selected_columns)
            self.logger.info(f"Selected {self.sensor_count} sensors")

            # Update each patient's dataset to adjust selected sensors based on above processes
            self.sensor_selection_trf = ps.ApplySensorSelection(selection=self.selected_columns)
            self.data_maps = {l_p_s_t_tuple: self.sensor_selection_trf.transform(mat_d)
                              for l_p_s_t_tuple, mat_d in tqdm(self.data_maps.items(),
                                                               desc='Applying sensor selection')}

            # -
            # Create a DataFrame to store mapping and labels to all windows identified across participant data maps
            sample_ix_df_l = list()
            # Specify types to keep dataframe size manageable - ORDER MATTERS
            key_col_dtypes = {'label': 'int8', 'sample_ix': 'int32', 'location': 'string',
                              'patient': 'int8', 'session': 'int8', 'trial': 'int8'}
            key_cols = list(key_col_dtypes.keys())

            for l_p_s_t, index_map in self.sample_index_maps.items():
                self.logger.info(f"Processing participant {l_p_s_t} index, having keys: {list(index_map.keys())}")
                # Pull out the dictionary results of pipeline outputs for this patients
                _data_map = self.data_maps[l_p_s_t]
                # Determine columns
                cols = key_cols + ['start_t', 'stop_t', 'indices']
                key_l = list(l_p_s_t)

                # Unpack a list of data tuples that will create a Dataframe
                # TODO: minimize mem footprint - can we generate a typed structured numpy arr, then from_records()?
                patient_ixes = [tuple([label_code, ix_i] + key_l + [_ix.min(), _ix.max(), _ix])
                                for label_code, indices_l in index_map.items()
                                for ix_i, _ix in enumerate(indices_l)]
                p_ix_df = pd.DataFrame(patient_ixes, columns=cols)
                p_ix_df = p_ix_df.astype(key_col_dtypes)

                # Store a numeric index into underlying numpy array for faster indexing
                if 'signal' in _data_map:
                    signal_df = _data_map['signal']
                    p_ix_df['start_ix'] = signal_df.index.get_indexer(p_ix_df.start_t)

                # #TODO: Is this still necessary? Determining the sentence code for every window sample from scratch
                if 'word_start_stop_times' in self.data_maps[l_p_s_t]:
                    self.logger.info(f"word_start_stop_times found - aligning all index start times to a sentence code")
                    wsst_df = self.data_maps[l_p_s_t]['word_start_stop_times']
                    nearest_ixes = wsst_df.index.get_indexer(p_ix_df.start_t, method='nearest')
                    p_ix_df['sent_code'] = wsst_df.iloc[nearest_ixes].stim_sentcode.values

                sample_ix_df_l.append(p_ix_df)

            self.logger.info(f"Combining all of {len(sample_ix_df_l)} index frames")
            self.sample_ix_df = pd.concat(sample_ix_df_l).reset_index(drop=True)
            self.k_select_offset = 2
            if self.flatten_sensors_to_samples:
                self.logger.info(f"flatten_sensors_to_samples selected - creating channel/sensor labels for samples")
                # Channel column will be exploded on to lengthen dataframe by X self.selected_columns
                self.sample_ix_df['channel'] = [self.selected_columns] * len(self.sample_ix_df)
                # This will add a new field that uniquely identifies a sample (the 'channel')
                key_cols.insert(2, 'channel')
                self.k_select_offset += 1
                self.logger.debug("exploding sensor data - does this take a while?")
                # Repeat rows by unique values in the list of channels in each row
                self.sample_ix_df = self.sample_ix_df.explode('channel')

            self.key_cols = key_cols

            # Auto make a reindex map from a reindex_col's every unique value
            if self.label_reindex_col is not None:
                self.label_reindex_ix = self.key_cols.index(self.label_reindex_col)
            elif self.label_reindex_col is None and self.label_reindex_map is not None:
                raise ValueError("label_reindex_col is required when label_reindex_map is provided")

            if self.label_reindex_map is None and self.label_reindex_col is not None:
                unique_reindex_labels = list(sorted(self.sample_ix_df[self.label_reindex_col].unique()))
                self.label_reindex_map = {l: i for i, l in enumerate(unique_reindex_labels)}

            self.logger.info("Converting dataframe to a flat list of key variables (self.flat_keys)")
            self.ixed_sample_ix_df = self.sample_ix_df.set_index(key_cols).sort_index()
            #key_df = self.sample_ix_df[self.key_cols]
            self.flat_keys = self.ixed_sample_ix_df.index
            #self.flat_keys = np.array(list(zip(key_df.to_records(index=False).tolist(),
            #                                   key_df.iloc[:, k_select_offset:].to_records(index=False).tolist())),
            #                          dtype='object')
            self.logger.info(f"Extracting mapping of ({key_cols})->indices")
            self.flat_index_map = self.ixed_sample_ix_df.indices#.to_dict()
            self.flat_ix_map = self.ixed_sample_ix_df.start_ix

            # ## END NEW VERSION

            self.logger.info(f"Length of flat index map: {len(self.flat_index_map)}")


        else:
            # print("Warning: using naive shared-referencing across objects - only use when feeling lazy")
            self.logger.warning("Warning: using naive shared-referencing across objects - only use when feeling lazy")
            # self.mfcc_m = self.data_from.mfcc_m
            self.data_maps = self.data_from.data_maps
            self.n_samples_per_window = self.data_from.n_samples_per_window
            self.sample_index_maps = self.data_from.sample_index_maps
            self.flat_index_map = self.data_from.flat_index_map
            self.flat_ix_map = self.data_from.flat_ix_map
            self.flat_keys = self.data_from.flat_keys
            self.key_cols = self.data_from.key_cols
            self.k_select_offset = self.data_from.k_select_offset
            # self.logger.info("Copying over sample ix dataframe")
            self.sample_ix_df = self.data_from.sample_ix_df.copy()
            self.ixed_sample_ix_df = self.data_from.ixed_sample_ix_df.copy()
            self.selected_columns = self.data_from.selected_columns
            self.flatten_sensors_to_samples = self.data_from.flatten_sensors_to_samples
            self.extra_output_keys = self.data_from.extra_output_keys
            self.fs_signal = self.data_from.fs_signal
            self.label_reindex_col = self.data_from.label_reindex_col
            self.label_reindex_ix = self.data_from.label_reindex_ix
            self.label_reindex_map = self.data_from.label_reindex_map

        self.select(self.selected_flat_indices)

    def make_pipeline_map(self, default='audio_gate'):
        """
        Pipeline parameters sometimes depend on the configuration of the dataset class,
        so for now it is bound method (not classmethod or staticmethod).
        """
        self.logger.debug(f"default pipeline: {default}")
        p_map = {
            'audio_gate': Pipeline([
                ('parse_signal', ps.ParseTimeSeriesArrToFrame(self.mat_d_keys['signal'],
                                                                    self.mat_d_keys['signal_fs'],
                                                                    default_fs=1200, output_key='signal')),
                ('parse_audio', ps.ParseTimeSeriesArrToFrame(self.mat_d_keys['audio'],
                                                                   self.mat_d_keys['audio_fs'],
                                                                   default_fs=48000, reshape=-1)),
                ('parse_stim', ps.ParseTimeSeriesArrToFrame(self.mat_d_keys['stimcode'],
                                                                  self.mat_d_keys['signal_fs'],
                                                                  default_fs=1200, reshape=-1, output_key='stim')),
                ('sensor_selection', ps.IdentifyGoodAndBadSensors(sensor_selection=self.sensor_columns)),
                ('subsample', ps.SubsampleSignal()),
                ('Threshold', ps.PowerThreshold(speaking_window_samples=48000 // 16,
                                                      silence_window_samples=int(48000 * 1.5),
                                                      speaking_quantile_threshold=0.9,
                                                      # n_silence_windows=5000,
                                                      # silence_threshold=0.001,
                                                      # silGence_quantile_threshold=0.05,
                                                      silence_n_smallest=5000)),
                ('speaking_indices', ps.WindowSampleIndicesFromStim('stim_pwrt',
                                                                          target_onset_shift=pd.Timedelta(-.5, 's'),
                                                                          # input are centers, and output is a window of .5 sec
                                                                          # so to center it, move the point (center) back .25 secods
                                                                          # so that extracted 0.5 sec window saddles the original center
                                                                          # target_offset_shift=pd.Timedelta(-0.25, 's')
                                                                          target_offset_shift=pd.Timedelta(-0.5, 's')
                                                                          )
                 ),

                ('silence_indices', ps.WindowSampleIndicesFromIndex('silence_stim_pwrt_s',
                                                                          # Center the extracted 0.5 second window
                                                                          index_shift=pd.Timedelta(-0.25, 's'),
                                                                          stim_value_remap=0
                                                                          )),
                ('output', 'passthrough')
            ]),

            #'minimal':
            #    feature_processing.SubsampleECOG() >>
            #    feature_processing.WordStopStartTimeMap() >> feature_processing.ChangSampleIndicesFromStim()
        }
        p_map['default'] = p_map[default]

        return p_map

    def to_eval_replay_dataloader(self, patient_k=None, data_k='ecog', stim_k='stim', win_step=1, batch_size=1024,
                                  num_workers=4,
                                  ecog_transform=None):
        if patient_k is None:
            patient_k = list(self.data_maps.keys())
        elif not isinstance(patient_k, list):
            patient_k = [patient_k]

        dl_map = dict()
        for k in patient_k:
            data_map = self.data_maps[k]
            ecog_torch_arr = torch.from_numpy(data_map[data_k].values).float()
            outputs = list()
            for _iix in tqdm(range(0, ecog_torch_arr.shape[0] - self.ecog_window_size, win_step),
                             desc='creating windows'):
                _ix = slice(_iix, _iix + self.ecog_window_size)
                feats = self.get_features(data_map, _ix, transform=ecog_transform, index_loc=True)
                # TODO: Just grabbing the max stim wode in the range - better or more useful way to do this?
                targets = self.get_targets(data_map, None, label=data_map['stim'].iloc[_ix].max())
                so = dict(**feats, **targets)
                so = {k: v for k, v in so.items()
                      if isinstance(v, torch.Tensor)}
                outputs.append(so)
            t_dl = torch.utils.data.DataLoader(outputs, batch_size=batch_size, num_workers=num_workers)
            dl_map[k] = t_dl

        ret = dl_map
        # if len(ret) == 1:
        #    ret = list(dl_map.values())[0]
        return ret

    def __len__(self):
        return len(self.selected_flat_keys)

    def __getitem__(self, item):
        # ix_k includes the class and window id, and possibly sensor id if flattened
        # data_k specifies subject dataset in data_map (less granular than ix_k)
        #ix_k, data_k = self.selected_flat_keys[item]
        ix_k = self.selected_flat_keys[item]
        data_k = ix_k[self.k_select_offset:]
        data_d = self.data_maps[data_k]

        selected_channels = None
        if self.flatten_sensors_to_samples:
            selected_channels = [ix_k[2]]

        so = dict()

        #ix = self.flat_index_map.at[ix_k]
        ix = self.flat_ix_map.at[ix_k]
        ix = range(ix, ix+self.n_samples_per_window)
        so.update(
            self.get_features(data_d, ix,
                              ix_k, transform=self.transform,
                              channel_select=selected_channels,
                              index_loc=True,
                              extra_output_keys=self.extra_output_keys,
                              all_input_transform=self.all_input_transform)
        )

        so.update(
            self.get_targets(data_d, ix,
                             # get the 0-1 label from the value of the selected reindex value - or just grab the first (default)
                             label=self.label_reindex_map[ix_k[self.label_reindex_ix]] if self.label_reindex_ix is not None else ix_k[0],
                             target_transform=self.target_transform)
        )

        if self.post_processing_func is not None:
            so_updates = self.post_processing_func(so, **self.post_processing_kws)
            so.update(so_updates)

        # Return anything that is a Torch Tensor - the torch dataloader will handle
        # compiling multiple outputs for batch
        return {k: v for k, v in so.items()
                if isinstance(v, torch.Tensor)}

    def split_select_at_time(self, split_time: float):
        # split_time = 0.75
        from tqdm.auto import tqdm

        selected_keys_arr = self.flat_keys[self.selected_flat_indices]
        index_start_stop = [(self.flat_index_map.at[a[0]].min(), self.flat_index_map.at[a[0]].max())
                            for a in tqdm(selected_keys_arr)]
        split_time = max(a for a, b, in index_start_stop) * split_time if isinstance(split_time, float) else split_time
        left_side_indices, right_side_indices = list(), list()
        for a, b in index_start_stop:
            if a < split_time:
                left_side_indices.append(a)
            else:
                right_side_indices.append(b)

        left_side_indices = np.array(left_side_indices)
        right_side_indices = np.array(right_side_indices)

        left_dataset = self.__class__(data_from=self, selected_word_indices=left_side_indices)
        right_dataset = self.__class__(data_from=self, selected_word_indices=right_side_indices)

        return left_dataset, right_dataset

    def split_select_random_key_levels(self, keys=('patient', 'sent_code'), **train_test_split_kws):
        from sklearn.model_selection import train_test_split
        keys = list(keys) if isinstance(keys, tuple) else keys
        # In case we have already split - check for existing selected indices
        if getattr(self, 'selected_flat_indices') is None:
            self.selected_flat_indices = range(0, self.sample_ix_df.shape[0] - 1)

        # Init the unique levels
        levels: pd.DataFrame = self.sample_ix_df.iloc[self.selected_flat_indices][keys].drop_duplicates()
        stratify_col = train_test_split_kws.get('stratify')
        if stratify_col is not None:
            train_test_split_kws['stratify'] = levels[stratify_col]

        # Split on the unique levels
        train, test = train_test_split(levels, **train_test_split_kws)
        self.logger.info(f"{len(levels)} levels in {keys} split into train/test")
        self.logger.info(f"Train: {train}")
        self.logger.info(f"Test : {test}")

        # Merge back to the original full sample_ix_df to determine the original index into the sample data
        train_indices = self.sample_ix_df[keys].reset_index().merge(train, on=keys, how='inner').set_index('index').index.tolist()
        test_indices = self.sample_ix_df[keys].reset_index().merge(test, on=keys, how='inner').set_index('index').index.tolist()

        # Create new train and test datsets - tack on the levels df for debugging, probably don't depend on them?
        train_dataset = self.__class__(data_from=self, selected_flat_indices=train_indices)
        train_dataset.selected_levels_df = train
        test_dataset = self.__class__(data_from=self, selected_flat_indices=test_indices)
        test_dataset.selected_levels_df = test

        return train_dataset, test_dataset

    def select(self, sample_indices):
        # select out specific samples from the flat_keys array if selection passed
        # - Useful if doing one-subject training and want to split data up among datasets for use
        self.selected_flat_indices = sample_indices
        if self.selected_flat_indices is not None:
            self.selected_flat_keys = self.flat_keys[self.selected_flat_indices]
        else:
            self.selected_flat_keys = self.flat_keys

        return self

    def append_transform(self, transform, all_input_transform: bool = False):
        transform = [transform] if not isinstance(transform, list) else transform
        if all_input_transform:
            self.all_input_transform_l += transform
            self.all_input_transform = torchvision.transforms.Compose(self.all_input_transform_l)
        else:
            self.transform_l += transform
            self.transform = torchvision.transforms.Compose(self.transform_l)
        return self

    def append_target_transform(self, transform):
        self.target_transform_l.append(transform)
        self.target_transform = torchvision.transforms.Compose(self.target_transform_l)
        return self

    ######
    @classmethod
    def load_mat_keys_from_path(cls, p):
        """
        Returns only the keys in the HDF5 file without loading the data
        """
        import h5py
        with h5py.File(p, 'r') as f:
            keys = list(f.keys())
        return keys

    @classmethod
    def load_mat_from_path(cls, p):
        """
        Loads all keys in HDF5 file into dict, convert values to np.array
        """
        try:
            mat_dat_map = scipy.io.loadmat(p)
        except NotImplementedError as e:
            msg = f"Couldn't load {os.path.split(p)[-1]} with scipy (vers > 7.3?) - using package 'mat73' to load"
            cls.logger.info(msg)

            import mat73
            mat_dat_map = mat73.loadmat(p)
        return mat_dat_map

    @classmethod
    def make_filename(cls, patient, session, trial, location):
        raise NotImplementedError()

    @classmethod
    def get_data_path(cls, patient, session, trial, location,
                      subset=None, base_path=None):
        fname = cls.make_filename(patient, session, trial, location)
        base_path = cls.default_base_path if base_path is None else base_path
        subset = cls.default_data_subset if subset is None else subset
        p = os.path.join(base_path, location, subset, fname)
        return p

    #######
    # Entry point to get data
    @classmethod
    def load_data(cls, location=None, patient=None, session=None, trial=None, base_path=None,
                  sensor_columns=None, subset=None):

        location = cls.default_location if location is None else location
        patient = cls.default_patient if patient is None else patient
        session = cls.default_session if session is None else session
        trial = cls.default_trial if trial is None else trial
        sensor_columns = cls.default_sensor_columns if sensor_columns is None else sensor_columns

        cls.logger.info(f"-----------Subset: {str(subset)}------------")
        cls.logger.info(f"---{patient}-{session}-{trial}-{location}---")

        p = cls.get_data_path(patient, session, trial, location, base_path=base_path, subset=subset)
        cls.logger.debug(f"Path : {p}")

        mat_d = cls.load_mat_from_path(p)
        cls.logger.debug(f"Matlab keys : {list(mat_d.keys())}")

        return mat_d

    @classmethod
    def make_tuples_from_sets_str(cls, sets_str):
        """
        Process a string representation of the patient tuples, e.g.: 'MC-19-0,MC-19-1'
        """
        if sets_str is None:
            return None

        # Select everything from all locations
        if sets_str.strip() == '*':
            return [t for loc, p_t_d in cls.all_patient_maps.items()
                     for t_l in p_t_d.values()
                     for t in t_l]

        # e.g. MC-19-0,MC-19-1
        if ',' in sets_str:
            sets_str_l = sets_str.split(',')
            # Recurse - returns a list, so combine all lists into one with `sum` reduction
            return sum([cls.make_tuples_from_sets_str(s) for s in sets_str_l], list())

        if '~' == sets_str[0]:
            return cls.make_all_tuples_with_one_left_out(sets_str[1:])

        set_terms = sets_str.split('-')
        # e.g. MC-22-1 has three terms ('MC', 22, 1) selecting a specific trial of a specific participant
        if len(set_terms) == 3:
            # org, pid, ix = sets_str.split('-')
            org, pid, ix = set_terms
            assert pid.isdigit() and ix.isdigit() and org in cls.all_patient_maps.keys()
            pmap, pid, ix = cls.all_patient_maps[org], int(pid), int(ix)
            assert pid in pmap, f"PID: {pid} not in {org}'s known data map"
            p_list = [pmap[pid][ix]]
        # e.g. MC-22 will return tuples for all of MC-22's data
        elif len(set_terms) == 2:
            org, pid = set_terms

            assert pid.isdigit(), f"pid expected to be a digit, but got {pid}"
            assert org in cls.all_patient_maps.keys(), f"org expected to be one of {list(cls.all_patient_maps.keys())}, but got {org}"

            pmap, pid = cls.all_patient_maps[org], int(pid)
            assert pid in pmap, f"PID: {pid} not in {org}'s known data map"
            p_list = pmap[pid]
        else:
            raise ValueError(f"Don't understand the {len(set_terms)} terms: {set_terms}")

        return p_list

    @classmethod
    def make_all_tuples_with_one_left_out(cls, sets_str):
        selected_t_l = cls.make_tuples_from_sets_str(sets_str)
        remaining_t_l = sum((l for pid_to_t_l in cls.all_patient_maps.values() for l in pid_to_t_l.values() if
                             all(o not in selected_t_l for o in l)),
                            start=list())
        return remaining_t_l

    @classmethod
    def make_remaining_tuples_from_selected(cls, sets_str):
        return list(set(cls.make_tuples_from_sets_str('*')) - set(cls.make_tuples_from_sets_str(sets_str)))

    @staticmethod
    def get_features(data_map, ix, label=None, transform=None, index_loc=False, signal_key='signal',
                     channel_select=None, extra_output_keys=None, all_input_transform=None):
        # pull out signal and begin building dictionary of arrays to reutrn
        signal_df = data_map[signal_key]

        kws = dict()
        kws['signal'] = signal_df.loc[ix].values if not index_loc else signal_df.values[ix]

        # Transpose to keep time as last index for torch
        #np_ecog_arr = kws['signal'].values.T
        np_ecog_arr = kws['signal'].T

        # if self.flatten_sensors_to_samples:
        # Always pass a list/array for channels, even if only 1, to maintain the dimension
        if channel_select is not None:
            np_ecog_arr = np_ecog_arr[channel_select]  # [None, :]

        if transform is not None:
            # print("Apply transform to shape of " + str(np_ecog_arr.shape))
            np_ecog_arr = transform(np_ecog_arr)

        kws['signal_arr'] = torch.from_numpy(np_ecog_arr).float()

        # extra_output_keys = ['sensor_ras_coord_arr']
        extra_output_keys = [extra_output_keys] if isinstance(extra_output_keys, str) else extra_output_keys
        if isinstance(extra_output_keys, list):
            kws.update({k: torch.from_numpy(data_map[k]).float() if isinstance(data_map[k], np.ndarray) else data_map[k]
                        for k in extra_output_keys})

            if 'sensor_ras_coord_arr' in kws and channel_select is not None:
                #                print(channel_select)
                #                if not isinstance(channel_select[0], int):
                #                    print("WHAT")
                kws['sensor_ras_coord_arr'] = kws['sensor_ras_coord_arr'][channel_select].unsqueeze(0)

        if all_input_transform is not None:
            # print("Apply transform to shape of " + str(np_ecog_arr.shape))
            kws = all_input_transform(kws)

        return kws

    def get_feature_shape(self):
        # TODO: Don't hardode signal array everywhere
        return self[0]['signal_arr'].shape

    @staticmethod
    def get_targets(data_map, ix, label, target_transform=None, target_key='target_arr'):
        #label = label[0]
        #kws = dict(text='<silence>' if label <= 0 else '<speech>',
        #           text_arr=torch.Tensor([0] if label <= 0 else [1]))
        kws = {target_key: torch.LongTensor([label])}
        if target_transform is not None:
            kws[target_key] = target_transform(kws[target_key])
        return kws

    @staticmethod
    def get_targets_old(data_map, ix, label, target_transform=None):
        label = label[0]

        kws = dict(text='<silence>' if label <= 0 else '<speech>',
                   text_arr=torch.Tensor([0] if label <= 0 else [1]))
        if target_transform is not None:
            kws['text_arr'] = target_transform(kws['text_arr'])
        return kws

    def sample_plot(self, i, band=None,
                    offset_seconds=0,
                    figsize=(15, 10), axs=None):
        import matplotlib
        from matplotlib import pyplot as plt
        from IPython.display import display
        # offs = pd.Timedelta(offset_seconds)
        # t_word_ix = self.word_index[self.word_index == i].index
        ix_k, data_k = self.selected_flat_keys[i]
        t_word_ix = self.flat_index_map.at[ix_k]
        offs_td = pd.Timedelta(offset_seconds, 's')
        t_word_slice = slice(t_word_ix.min() - offs_td, t_word_ix.max() + offs_td)
        display(t_word_slice)
        display(t_word_ix.min() - offs_td)
        # t_word_ix = self.word_index.loc[t_word_ix.min() - offs_td: t_word_ix.max() - offs_td].index
        # t_word_ecog_df = self.ecog_df.reindex(t_word_ix).dropna()
        # t_word_wav_df = self.speech_df.reindex(t_word_ix)
        ecog_df = self.data_maps[data_k]['signal']
        speech_df = self.data_maps[data_k]['audio']
        word_txt = "couldn't get word or text mapping from data_map"
        if 'word_code_d' in self.data_maps[data_k]:
            word_txt = self.data_maps[data_k]['word_code_d'].get(ix_k[0], '<no speech>')

        t_word_ecog_df = ecog_df.loc[t_word_slice].dropna()
        t_word_wav_df = speech_df.loc[t_word_slice]
        # display(t_word_ecog_df.describe())
        # scols = self.default_sensor_columns
        scols = self.selected_columns

        ecog_std = ecog_df[scols].std()
        cmap = matplotlib.cm.viridis_r
        norm = matplotlib.colors.Normalize(vmin=ecog_std.min(), vmax=ecog_std.max())
        c_f = lambda v: cmap(norm(v))
        colors = ecog_std.map(c_f).values

        if axs is None:
            fig, axs = plt.subplots(figsize=figsize, nrows=2)
        else:
            fig = axs[0].get_figure()

        if band is not None:
            plt_df = t_word_ecog_df[scols].pipe(proc.filter, band=band,
                                                sfreq=self.fs_signal)
        else:
            plt_df = t_word_ecog_df[scols]

        ax = plt_df.plot(alpha=0.3, legend=False,
                         color=colors, lw=1.2,
                         ax=axs[0], fontsize=14)
        ax.set_title(f"{len(plt_df)} samples")

        ax = t_word_wav_df.plot(alpha=0.7, legend=False, fontsize=14, ax=axs[1])
        ax.set_title(f"{len(t_word_wav_df)} samples, word = {word_txt}")

        fig.tight_layout()

        return axs

    def get_target_shape(self):#, target_key='target_arr'):
        if self.label_reindex_col is None:
            n_targets = self.sample_ix_df.label.nunique()
        else:
            # label_reindex_map is key->class_id, so number of unique values in the map (as opposed to its keys)
            n_targets = len(set(self.label_reindex_map.values()))
        # handle special case for binary
        return 1 if n_targets == 2 else n_targets

    def get_target_labels(self):
        # TODO: Warning - assuming these are all the same across data_maps values - just using the first
        if self.label_reindex_col is None:
            class_val_to_label_d = next(iter(self.data_maps.values()))['index_source_map']
        else:
            class_val_to_label_d = dict()
            for label, cls_id in self.label_reindex_map.items():
                class_val_to_label_d[cls_id] = class_val_to_label_d.get(cls_id, []) + [label]

            class_val_to_label_d = {cls_id: "_".join(map(str, label_l))
                                    for cls_id, label_l in class_val_to_label_d.items()}
            #class_val_to_label_d = {cls_id: f"{self.label_reindex_col}_{label}"
            #                        for label, cls_id in self.label_reindex_map.items()}

        class_labels = [class_val_to_label_d[i] for i in range(len(class_val_to_label_d))]
        return class_val_to_label_d, class_labels


