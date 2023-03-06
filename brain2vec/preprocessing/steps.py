from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import attr
from mmz import utils
import torch
from typing import Optional

with_logger = utils.with_logger(prefix_name=__name__)


@attr.s
@with_logger
class DictTrf(BaseEstimator, TransformerMixin):
    def transform(self, data_map):
        in_keys = set(data_map.keys())
        updates = self.process(data_map)
        out_keys = set(updates.keys())
        self.logger.debug(f"Updated keys: {out_keys}")
        data_map.update(updates)
        return data_map

    def process(self, data_map):
        raise NotImplementedError()


@attr.s
@with_logger
class ParseTimeSeriesArrToFrame(DictTrf):
    array_key = attr.ib()
    fs_key = attr.ib()
    default_fs = attr.ib()
    slice_selection = attr.ib(None)
    dtype = attr.ib(None)
    reshape = attr.ib(None)
    output_key = attr.ib(None)

    def process(self, data_map):
        # determine what the output key will be
        arr_key = self.array_key if self.output_key is None else self.output_key

        try:
            self.logger.debug(f"Accessing {self.fs_key}")
            _fs = data_map[self.fs_key]
        except KeyError as ke:
            msg = f"provided fs_key={self.fs_key} not in data_map, expected one of: {list(data_map.keys())}"
            self.logger.warning(msg)
            _fs = self.default_fs

        self.logger.debug(f"Input source frequency, fs object: {_fs}")

        # Sometimes it's a scalar value inside some arrays
        if isinstance(_fs, np.ndarray):
            fs = data_map[self.fs_key].reshape(-1)[0]
        else:
            # Otherwise, just make sure it is integer
            fs = int(_fs)

        arr = data_map[self.array_key]

        if self.reshape is not None:
            arr = arr.reshape(self.reshape)

        ix = pd.TimedeltaIndex(pd.RangeIndex(0, arr.shape[0]) / fs, unit='s')
        if arr.ndim == 1:
            arr_df = pd.Series(arr, index=ix, dtype=self.dtype, name=arr_key).sort_index()
        else:
            arr_df = pd.DataFrame(arr, index=ix, dtype=self.dtype).sort_index()

        if self.slice_selection is not None:
            arr_df = arr_df.loc[self.slice_selection].copy()

        self.logger.info(f"{self.array_key}@{fs}, shape: {arr_df.shape}, [{arr_df.index[0], arr_df.index[-1]}]")
        assert arr_df.index.is_unique, f"NON UNIQUE TIME SERIES INDEX FOR KEY {self.array_key}"

        return {self.fs_key: fs, arr_key: arr_df}


@attr.s
@with_logger
class IdentifyGoodAndBadSensors(DictTrf):
    electrode_qual_key = attr.ib('electrodes')
    on_missing = attr.ib('ignore')
    sensor_selection = attr.ib(None)
    good_electrode_ind_column = attr.ib(0)

    def process(self, data_map):
        k = self.electrode_qual_key
        if k not in data_map:
            msg = f"Electrodes with key '{self.electrode_qual_key}' not found among {list(data_map.keys())}"
            if self.on_missing == 'ignore':
                self.logger.warning(msg + ' - but on_missing="ignore", so moving on')
                return dict(good_sensor_columns=None, bad_sensor_columns=None)
            else:
                raise KeyError("ERROR: " + msg)

        chann_code_cols = ["code_%d" % e for e in range(data_map[k].shape[-1])]
        channel_df = pd.DataFrame(data_map['electrodes'], columns=chann_code_cols)
        self.logger.info("Found N electrodes = %d" % channel_df.shape[0])

        # required_sensor_columns = channel_df.index.tolist() if sensor_columns is None else sensor_columns
        # Mask for good sensors
        ch_m = (channel_df.iloc[:, self.good_electrode_ind_column] == 1)
        all_valid_sensors = ch_m[ch_m].index.tolist()

        # Spec the number of sensors that the ecog array mush have
        if self.sensor_selection is None:
            required_sensor_columns = channel_df.index.tolist()
        elif self.sensor_selection == 'valid':
            sensor_columns = all_valid_sensors
            required_sensor_columns = sensor_columns
        else:
            required_sensor_columns = self.sensor_selection

        #
        good_sensor_columns = [c for c in all_valid_sensors if c in required_sensor_columns]
        bad_sensor_columns = list(set(required_sensor_columns) - set(good_sensor_columns))
        return dict(good_sensor_columns=good_sensor_columns, bad_sensor_columns=bad_sensor_columns,
                    channel_status=channel_df)


@attr.s
@with_logger
class ApplySensorSelection(DictTrf):
    selection = attr.ib(None)
    signal_key = attr.ib('signal')
    bad_sensor_method = attr.ib('zero')

    @classmethod
    def select_from_ras(cls, data_map, selected_columns, bad_columns, **kws):
        # s_df = data_map['sensor_ras_df'].loc[selected_columns]
        s_df = data_map['sensor_ras_df'].loc[selected_columns]
        return {'sensor_ras_df': s_df,
                'sensor_ras_coord_arr': s_df.filter(like='coord').values
                }

    def process(self, data_map):
        # TODO: This doesn't actually select any sensors?
        signal_df = data_map[self.signal_key]
        bs_cols = list()
        missing_cols = list()
        selected_cols = None

        if self.selection is None:
            self.logger.info("Checking data_map for good_sensor_columns")
            selected_cols = data_map.get('good_sensor_columns')
            self.logger.inf(f"Got: {selected_cols}")
        elif isinstance(self.selection, list):
            self.logger.info(f"Selection of columns passed to sensor selection")
            selected_cols = self.selection
        else:
            raise ValueError(f"Don't understand selection: {self.selection}")

        if selected_cols is None:
            selected_cols = signal_df.columns.tolist()
            missing_cols = list()
        else:
            missing_cols = [c for c in selected_cols if c not in signal_df.columns]
            # missing_cols = signal_df.columns[~signal_df.columns.isin(selected_cols)].tolist()

        bs_cols += data_map['bad_sensor_columns'] if data_map['bad_sensor_columns'] is not None else list()
        bs_cols = bs_cols + missing_cols

        sel_signal_df = signal_df.copy()
        if bs_cols is not None and len(bs_cols) > 0:
            if self.bad_sensor_method == 'zero':
                self.logger.info(f"ZEROING columns that were missing or bad")
                self.logger.info(f"\tMissing: {missing_cols}")
                self.logger.info(f"\tBad: {data_map['bad_sensor_columns']}")
                sel_signal_df.loc[:, bs_cols] = 0.
            elif self.bad_sensor_method == 'ignore':
                self.logger.warning(f"Ignoring bad sensor columns: {bs_cols}")
            else:
                raise KeyError(f"Don't understand bad_sensor_method: {self.bad_sensor_method}")

        r_val = {self.signal_key: sel_signal_df[selected_cols],
                 'selected_columns': selected_cols,
                 'bad_columns': bs_cols,
                 'bad_sensor_method': self.bad_sensor_method}

        # TODO: a way to apply a bunch of selection functions
        if 'sensor_ras_df' in data_map:
            self.logger.info("Selecting columns in RAS coordinate data")
            self.logger.info(f"Bad columns {bs_cols}")
            sensor_ras_df = data_map['sensor_ras_df']

            bs_ras_df = pd.DataFrame([dict(electrode_name='UNKW', contact_number=contact_num,
                                           x_coord=0., y_coord=0., z_coord=0.) for contact_num in bs_cols])
            # sensor_ras_df.loc[bs_cols, :] = 0
            sensor_ras_df = pd.concat([sensor_ras_df, bs_ras_df]).reset_index(drop=True)
            data_map['sensor_ras_df'] = sensor_ras_df
            ras_sel = self.select_from_ras(data_map, **r_val)
            r_val.update(ras_sel)

        return r_val


@attr.s
@with_logger
class SubsampleSignal(DictTrf):
    signal_keys = attr.ib(('signal', 'stim'))
    signal_rate_key = attr.ib('fs_signal')
    rate = attr.ib(2)

    def process(self, data_map):
        output = {k: data_map[k].iloc[::self.rate] for k in self.signal_keys}
        output[self.signal_rate_key] = int((1. / self.rate) * data_map[self.signal_rate_key])
        return output


@attr.s
@with_logger
class CreateAllStim(DictTrf):
    """A stim that is always true, to use when needing a 'default' mask across all time samples"""
    stim_key = attr.ib('stim')

    def process(self, data_map):
        s = data_map[self.stim_key]
        return dict(all_stim=pd.Series(True, index=s.index, name='all_stim'))


@attr.s
@with_logger
class ReplaceSignalWithPinkNoise(DictTrf):
    signal_key = attr.ib('signal')
    signal_rate_key = attr.ib('fs_signal')
    output_key = attr.ib(None)

    def process(self, data_map):
        import pyplnoise
        signal = data_map[self.signal_key]
        fs_signal = data_map[self.signal_rate_key]
        output_key = self.signal_key if self.output_key is None else self.output_key
        pknoise = pyplnoise.PinkNoise(fs_signal, 1e-2, fs_signal // 2)
        from tqdm.auto import tqdm

        pk_df = pd.DataFrame({k: pknoise.get_series(signal.shape[0]).astype('float32') for k in
                              tqdm(range(signal.shape[1]), desc='Generating PinkNoise')})

        if isinstance(signal, pd.DataFrame):
            pk_df.columns = signal.columns
            pk_df.index = signal.index

        return {output_key: pk_df}


@attr.s
@with_logger
class StandardNormSignal(DictTrf):
    signal_key = attr.ib('signal')
    output_key = attr.ib('signal')
    rate = attr.ib(2)

    def process(self, data_map):
        df = data_map[self.signal_key]
        mu = df.mean()
        std = df.std()
        return {self.output_key: (df - mu) / std}


@attr.s
@with_logger
class ExtractMFCC(DictTrf):
    n_fft = attr.ib(1024)
    win_length = attr.ib(None)
    hop_length = attr.ib(512)
    n_mels = attr.ib(13)
    fs = attr.ib(None)

    audio_key = attr.ib('audio')
    audio_fs_key = attr.ib('fs_audio')

    def process(self, data_map):
        audio_s = data_map[self.audio_key]
        fs = data_map[self.audio_fs_key]

        if not hasattr(self, 'melspec_trf'):
            import torchaudio

            self.melspec_trf = torchaudio.transforms.MelSpectrogram(fs,
                                                                    n_fft=self.n_fft,
                                                                    win_length=self.win_length,
                                                                    hop_length=self.hop_length,
                                                                    center=True,
                                                                    pad_mode="reflect",
                                                                    power=2.0,
                                                                    norm="slaney",
                                                                    onesided=True,
                                                                    n_mels=self.n_mels,
                                                                    mel_scale="htk")

        audio_arr = torch.from_numpy(audio_s.values).float()
        audio_mfc = self.melspec_trf(audio_arr)

        audio_mfc_df = pd.DataFrame(audio_mfc.T.detach().cpu())
        mfc_ix = pd.TimedeltaIndex(pd.RangeIndex(0, audio_mfc_df.shape[0]) / (fs / self.melspec_trf.hop_length),
                                   unit='s')
        audio_mfc_df.index = mfc_ix

        return dict(audio_mel_spec=audio_mfc_df)


@attr.s
@with_logger
class PowerThreshold(DictTrf):
    stim_key = attr.ib('stim')
    audio_key = attr.ib('audio')

    speaking_threshold = attr.ib(0.005)
    silence_threshold = attr.ib(0.002)

    speaking_window_samples = attr.ib(48000)
    # More silence data, so require larger region of threshold check
    silence_window_samples = attr.ib(48000)
    stim_silence_value = attr.ib(0)
    silence_quantile_threshold = attr.ib(None)
    silence_n_smallest = attr.ib(None)
    n_silence_windows = attr.ib(35000)
    speaking_quantile_threshold = attr.ib(None)

    def process(self, data_map):
        return self.power_threshold(data_map[self.audio_key], data_map[self.stim_key],
                                    speaking_threshold=self.speaking_threshold,
                                    speaking_window_samples=self.speaking_window_samples,
                                    silence_threshold=self.silence_threshold,
                                    silence_window_samples=self.silence_window_samples,
                                    stim_silence_value=self.stim_silence_value,
                                    n_silence_windows=self.n_silence_windows,
                                    speaking_quantile_threshold=self.speaking_quantile_threshold,
                                    silence_quantile_threshold=self.silence_quantile_threshold,
                                    silence_n_smallest=self.silence_n_smallest)

    @classmethod
    def power_threshold(cls, audio_s, stim_s, speaking_threshold,
                        speaking_window_samples,
                        silence_threshold,
                        silence_window_samples, stim_silence_value,
                        n_silence_windows,
                        speaking_quantile_threshold,
                        silence_quantile_threshold,
                        silence_n_smallest):
        cls.logger.info("Power threshold")
        #### Speaking
        rolling_pwr = (audio_s
                       .abs().rolling(speaking_window_samples, center=True)
                       .median().reindex(stim_s.index, method='nearest').fillna(0))
        # .max().reindex(stim_s.index, method='nearest').fillna(0))

        if speaking_quantile_threshold is not None:
            cls.logger.info(f"Using speaking quantile {speaking_quantile_threshold}")
            speaking_quantile_threshold = float(speaking_quantile_threshold)
            thresholded_speaking_pwr = rolling_pwr.pipe(lambda s: s > s[s > 0].quantile(speaking_quantile_threshold))
        else:
            thresholded_speaking_pwr = (rolling_pwr > speaking_threshold)

        speaking_stim_auto_m = (stim_s != stim_silence_value) & thresholded_speaking_pwr

        #### Silence
        silence_rolling_pwr = (audio_s
                               .abs().rolling(silence_window_samples, center=True)
                               .median().reindex(stim_s.index, method='nearest').fillna(np.inf))
        # .max().reindex(stim_s.index, method='nearest').fillna(0))

        if silence_n_smallest is not None:
            silence_n_smallest = int(silence_n_smallest)
            cls.logger.info(f"Using silence {silence_n_smallest} smallest on {type(silence_rolling_pwr)}")
            n_smallest = silence_rolling_pwr.nsmallest(silence_n_smallest)
            n_smallest_ix = n_smallest.index
            # cls.logger.info(f"n smallest: {n_smallest_ix.()}")
            thresholded_silence_pwr = pd.Series(False, index=silence_rolling_pwr.index)
            thresholded_silence_pwr.loc[n_smallest_ix] = True
        elif silence_quantile_threshold is not None:
            cls.logger.info("Using silence quantile")
            silence_quantile_threshold = float(silence_quantile_threshold)
            thresholded_silence_pwr = silence_rolling_pwr.pipe(lambda s: s <= s.quantile(silence_quantile_threshold))
        else:
            cls.logger.info("Using silence power-threshold")
            thresholded_silence_pwr = (silence_rolling_pwr < silence_threshold)

        # silence_stim_auto_m = (stim_s == stim_silence_value) & (~speaking_stim_auto_m) & thresholded_silence_pwr
        # silence_stim_auto_m = (stim_s == stim_silence_value) & thresholded_silence_pwr
        silence_stim_auto_m = (~speaking_stim_auto_m) & thresholded_silence_pwr

        if n_silence_windows is not None and n_silence_windows > 0:
            available_silence_stim: float = silence_stim_auto_m.sum()
            cls.logger.info(f"Sampling {n_silence_windows} from {available_silence_stim}")
            kws = dict(replace=False)
            if n_silence_windows > available_silence_stim:
                cls.logger.warning(
                    "More silent stims requested than available (see above INFO) - sampling with replace")
                kws['replace'] = True
            silence_samples = silence_stim_auto_m[silence_stim_auto_m].sample(n_silence_windows, **kws)
            silence_stim_auto_m = pd.Series(False, index=silence_stim_auto_m.index)
            silence_stim_auto_m[silence_samples.index] = True

        # Is the number of unique word codes different when using the threshold selected subset we
        # just produced (stim_auto_m)?
        # - Subtract one for no speech (0)
        eq = (stim_s.nunique(dropna=False) - 1) == stim_s[speaking_stim_auto_m].nunique(dropna=False)

        if not eq:
            msg = "stim_s and stim_auto not equal: %d - 1 != %d" % (stim_s.nunique(False),
                                                                    stim_s[speaking_stim_auto_m].nunique(False))
            # print(msg)
            cls.logger.warning(msg)

        # Create a new stim array with original word code where it's set, otherwise zero
        stim_pwrt_s = pd.Series(np.where(speaking_stim_auto_m, stim_s, 0), index=stim_s.index)
        stim_pwrt_diff_s = stim_pwrt_s.diff().fillna(0).astype(int)

        silence_stim_pwrt_s = pd.Series(np.where(silence_stim_auto_m, 1, 0), index=stim_s.index)
        silence_stim_pwrt_diff_s = silence_stim_pwrt_s.diff().fillna(0).astype(int)

        # coded_silence_stim = (silence_stim_pwrt_diff_s.cumsum() + 1) * silence_stim_pwrt_s
        coded_silence_stim = (silence_stim_pwrt_s.diff().eq(-1).cumsum() + 1) * silence_stim_pwrt_s

        updates = dict(stim_pwrt=stim_pwrt_s, stim_pwrt_diff=stim_pwrt_diff_s,
                       silence_stim_pwrt_s=silence_stim_pwrt_s, silence_stim_pwrt_diff_s=silence_stim_pwrt_diff_s,
                       coded_silence_stim=coded_silence_stim,
                       rolling_audio_pwr=rolling_pwr)
        return updates


@attr.s
@with_logger
class SentCodeFromStartStopWordTimes(DictTrf):
    """
    Parses a "sentence" stim identifying an entire region of experiment activity from HVS-Style encoded stim.

    The sentence stim identifies the region of activity pertaining to a specific sentence. The
    resulting stim will be the sentence code in the original stim that is active only during the
    "listening" region of activity. But this output stim will extend to the entire region.
    """
    stim_speaking_value = attr.ib(51)
    stim_key = attr.ib('stim')

    @classmethod
    def parse_start_stop_word_ms(cls, sswms):
        word_df = pd.DataFrame(sswms, columns=['start_t', 'stop_t', 'word'])
        # convert to time in seconds - TODO: pipe and call on whole series? should be fast...
        word_df['start_t'] = word_df.start_t.astype(float).apply(lambda v: pd.Timedelta(v, 's'))
        word_df['stop_t'] = word_df.stop_t.astype(float).apply(lambda v: pd.Timedelta(v, 's'))

        return word_df

    def process(self, data_map):
        # Convert start stop times to a dataframe and set some types
        word_df = self.parse_start_stop_word_ms(data_map['start_stop_word_ms'])

        # Stim should be a step-like signal, with different values for different regions
        stim = data_map[self.stim_key]

        # speaking is lowest stim code - find all word codes for when they are listening (lt(stim_speaking))
        listening_stim_s = stim[stim.lt(self.stim_speaking_value) & stim.gt(0)]

        # Get the listening sample nearest to the words start time from the listening index
        # This is an INDEX (0, 1, 2, erx.), not timestamps
        start_listening_ixes = listening_stim_s.index.get_indexer(word_df.start_t.values, method='nearest')

        # Get the index nearest to the words start time for the stim values - should basically be the start_t value
        # This is an INDEX (0, 1, 2, erx.), not timestamps
        start_stim_ixes = stim.index.get_indexer(word_df.start_t, method='nearest')

        # stim_start_t is the stim's timestamp where the start time of the spoken word is
        # So this should essentially match the start_t column
        # Note that we are indexing into the index itself, not the values
        word_df['stim_start_t'] = stim.index[start_stim_ixes]

        # Get the filtered stim code (to only sentence/listening code) to the point closest to each row's start_t
        word_df['stim_sentcode'] = listening_stim_s.iloc[start_listening_ixes].values

        # Get the time of that stim code (sentence code) neared to the at the point closest to each rows start_t
        word_df['stim_sentcode_t'] = listening_stim_s.iloc[start_listening_ixes].index

        # add the stim in just cause - it should always be 51 (?) for speaking code since this is forced alignment data
        word_df = word_df.set_index('stim_start_t').join(stim)

        ### ----
        # NOTE: Check for repeated stim codes (..only seen in UCSD 28, sent code 45), this
        #       **adds sent code 45.5 to their stim**
        # The max time we expect a sentence code to be
        grps, max_delta = list(), pd.Timedelta(1, 'm')
        for sent_code, sc_df in word_df.groupby('stim_sentcode'):
            # Capture how long the segment is
            delta_t = sc_df.stim_sentcode_t.max() - sc_df.stim_sentcode_t.min()
            o_cs_df = sc_df

            if delta_t > max_delta:
                self.logger.warning(f"Sent code {sent_code} has a time range more than a minute: {delta_t}")
                # Limit to the onset markers and find the word start (speaking
                # start) with the biggest jump in sent code time
                split_point_t = sc_df[sc_df.word.eq('on')].stim_sentcode_t.diff().idxmax()
                # Grab the first instance, dropping the last sample that is actually from the second instance
                first_w_df = sc_df.loc[:split_point_t].iloc[:-1].copy()
                # Get the last stim sent code for the first instance of the duplicate sent code
                split_point_t = first_w_df.iloc[-1].stop_t
                # The latter portion of the word df, after the split
                last_w_df = sc_df.loc[split_point_t:].copy()
                # Give it a word code that does't exist, but clear where it came from , so + 0.5
                last_w_df['stim_sentcode'] = last_w_df['stim_sentcode'] + 0.5

                # WARNING: Change the stim - replace instances past the split point with the new code
                stim.loc[split_point_t:] = stim.loc[split_point_t:].replace(sent_code, sent_code + 0.5)

                # put it all together
                o_cs_df = pd.concat([
                    first_w_df,
                    last_w_df
                ])

            grps.append(o_cs_df)

        word_df = pd.concat(grps).sort_index()
        # END
        ### ----

        # Extract sentence level stim
        sent_df = pd.concat([word_df.groupby('stim_sentcode').start_t.min(),
                             word_df.groupby('stim_sentcode').stop_t.max()], axis=1)

        sent_df['length'] = sent_df.diff(axis=1).stop_t.rename('length')
        # sent_df = sent_df.join(sent_df.diff(axis=1).stop_t.rename('length'))

        return dict(word_start_stop_times=word_df,
                    sent_start_stop_time=sent_df,
                    stim=stim
                    )


@with_logger
@attr.s
class AppendExtraMultiTaskStartStop(DictTrf):
    """
    Update start_stop_times frame to include region start-stop, as well as <region>_word_start/stop_t, for each region

    Expects sentence code to exist in the input datamap
    """

    # listening_stim_range_tuple = attr.ib((1, 50))
    def process(self, data_map):
        wsst_df = data_map['word_start_stop_times'].copy()
        stim = data_map['stim'].copy()

        wsst_df['speaking_word_length_t'] = wsst_df.stop_t - wsst_df.start_t

        # Map each sentence code to a pd.Series of offset times from the first spoken word start time
        # So the first spoken word in the sentence will be at 0 seconds time delta
        sent_code_reffs_d = {sent_code: sent_df.start_t - sent_df.start_t.min()
                             for sent_code, sent_df in wsst_df.groupby('stim_sentcode')}
        # Stack the results back into a single Series and join it back into the word start stop time frame
        offset_from_start_s = pd.concat(sent_code_reffs_d.values(), axis=0).rename('time_from_speaking_start')
        wsst_df = wsst_df.join(offset_from_start_s)

        # Get a listening only stim
        # listening_sent_stim_s = stim[stim.between(*self.listening_stim_range_tuple)]
        listening_sent_stim_s = stim[stim.isin(wsst_df.stim_sentcode.unique())]
        # For each listening part of the stim, capture its overall region by identifying the listening stim's
        # start (minimum time) and the *next* listening stim's start
        trial_regions_l = list()
        for sent_code, sent_s in listening_sent_stim_s.groupby(listening_sent_stim_s):
            # Start time
            trial_start_t = sent_s.index.min()
            # ALl future listening stims
            future_listening_s = listening_sent_stim_s.loc[sent_s.index.max():]
            # All future listening stims not equal to this stim
            future_ne_s = future_listening_s[future_listening_s.ne(sent_code)]
            # If there are future stims, take their minimum times
            if len(future_ne_s) > 0:
                start_of_next_t = future_ne_s.index.min()
            # Otherwise, we are on the last region, just go to the end (max of future)
            else:
                self.logger.warning(f"Sent code {sent_code} has no future sentence codes in front of it")
                future_s = stim.loc[sent_s.index.max():]
                start_of_next_t = future_s.index.max()

            # trial_end_t = listening_sent_stim_s.loc[:start_of_next_t].index[-1]
            # The trials last sample is the last sample leading up to the start of next # TODO: is this inclusive?
            trial_end_t = stim.loc[:start_of_next_t].index[-1]
            trial_regions_l.append(dict(stim_sentcode=sent_code, trial_start_t=trial_start_t, trial_stop_t=trial_end_t))

        # Stack together with columns (see dict above): stim_sentcode, trial_start_t, and trial_stop t
        trial_regions_df = pd.DataFrame(trial_regions_l)
        regions_l = list()

        # TODO: unnecessary closure?
        def _mm_work(s):
            return {f'{s.name}_start_t': s.index.min(),
                    f'{s.name}_stop_t': s.index.max()}

        # Go through the trial regions and create
        for ix, r in trial_regions_df.iterrows():
            _s = stim.loc[r.trial_start_t: r.trial_stop_t]
            # _s = _s[_s.isin((r.stim_sentcode, 0, 51, 52, 53))]
            regions_l.append(dict(
                stim_sentcode=r.stim_sentcode,
                **_s[_s.eq(r.stim_sentcode)].rename('listening_region').pipe(_mm_work),
                **_s[_s.eq(51)].rename('speaking_region').pipe(_mm_work),
                **_s[_s.eq(52)].rename('imagining_region').pipe(_mm_work),
                **_s[_s.eq(53)].rename('mouthing_region').pipe(_mm_work),
            ))

        regions_df = pd.DataFrame(regions_l)

        wsst_df = wsst_df.merge(trial_regions_df, on='stim_sentcode').merge(regions_df, on='stim_sentcode')  #

        start_offs_s = wsst_df.time_from_speaking_start
        stop_offs_s = wsst_df.time_from_speaking_start + wsst_df.speaking_word_length_t
        wsst_df = wsst_df.assign(
            listening_word_start_t=wsst_df.listening_region_start_t + start_offs_s,
            listening_word_stop_t=wsst_df.listening_region_start_t + stop_offs_s,

            speaking_word_start_t=wsst_df.speaking_region_start_t + start_offs_s,
            speaking_word_stop_t=wsst_df.speaking_region_start_t + stop_offs_s,

            mouthing_word_start_t=wsst_df.mouthing_region_start_t + start_offs_s,
            mouthing_word_stop_t=wsst_df.mouthing_region_start_t + stop_offs_s,

            imagining_word_start_t=wsst_df.imagining_region_start_t + start_offs_s,
            imagining_word_stop_t=wsst_df.imagining_region_start_t + stop_offs_s,
        )

        wsst_df = wsst_df.set_index('start_t', drop=False)
        wsst_df.index.name = 'stim_start_t'

        return dict(word_start_stop_times=wsst_df)


@attr.s
@with_logger
class ParseSensorRAS(DictTrf):
    ras_key = attr.ib('label_contact_common')

    @staticmethod
    def ras_to_frame(ras):
        return pd.DataFrame(
            # Pass to numpy array first to have it coalesce all the scalar string arrays into a multi dim object arr
            np.array(ras),
            # hardcoded columns for now
            columns=['electrode_name', 'contact_number', 'x_coord', 'y_coord', 'z_coord']
            # Use Pandas to try and convert everything to numbers
        ).apply(pd.to_numeric, downcast='float', errors='ignore')

    def process(self, data_map):
        ras_df = self.ras_to_frame(data_map[self.ras_key])
        ras_arr = ras_df[['x_coord', 'y_coord', 'x_coord']].values
        return dict(sensor_ras_df=ras_df, sensor_ras_coord_arr=ras_arr)


@attr.s
@with_logger
class NewStimFromRegionStartStopTimes(DictTrf):
    start_t_column = attr.ib('start_t')
    stop_t_column = attr.ib('stop_t')
    label_column = attr.ib('word')
    code_column = attr.ib(None)
    group_code_column = attr.ib('stim_sentcode')

    stim_output_name = attr.ib('word_stim')

    start_stop_time_input_name = attr.ib('word_start_stop_times')
    series_with_timestamp_index = attr.ib('stim')
    default_stim_value = attr.ib(0)

    def process(self, data_map):
        _word_df = data_map[self.start_stop_time_input_name].copy()
        time_s = data_map[self.series_with_timestamp_index]
        ix = time_s.index
        t_cols = [self.start_t_column, self.stop_t_column]

        output_stim = pd.Series(self.default_stim_value, index=ix, name=self.stim_output_name)
        output_mask = pd.Series(False, index=ix, name=self.stim_output_name + '_mask')

        code_maps = list()
        working_ix = 0
        for i, (gname, gdf) in enumerate(_word_df.groupby(self.group_code_column)):
            # Spoken is word is all caps string and other fields are replicated across all words
            is_word_m = gdf.word.str.upper() == gdf.word
            # Drop duplicates on the times - in case regions are selected, that have repeating values, we only want
            # a single stim
            dd_word_df = gdf[is_word_m].drop_duplicates(subset=t_cols).sort_values(self.start_t_column)

            for ii, (ix, r) in enumerate(dd_word_df.iterrows()):
                _label_val = r[self.label_column]
                _start_t = r[self.start_t_column]
                _stop_t = r[self.stop_t_column]

                _start_i, _stop_i = output_stim.index.get_indexer([_start_t, _stop_t], method='nearest')

                _code = (working_ix := working_ix + 1) if self.code_column is None else r[self.code_column]
                code_maps.append({self.group_code_column: gname,
                                  self.label_column: _label_val,
                                  # self.word_code_map_output_name: _code,
                                  'start_t': _start_t})
                output_stim.iloc[_start_i: _stop_i] = _code
                output_mask.iloc[_start_i: _stop_i] = True

        out_d = {self.stim_output_name: output_stim.rename(self.stim_output_name),
                 self.stim_output_name + '_mask': output_mask
                 # self.sentence_stim_output_name: sentence_stim.rename(self.sentence_stim_output_name)
                 }

        return out_d


@attr.s
@with_logger
class SelectWordsFromStartStopTimes(DictTrf):
    selected_words = [  # 'HER',
        'TANK',
        'SUN',
        # 'SIZE',
        'FISH',
        'FENCE',
        # 'OVER',
        'BLUE',
        'DAYS',
        # 'LEFT',
        # 'BOY',
        'CLEAR',
        'YOUNG',
        # 'BEFORE',
        'GIRL',
        'GAVE'
    ]

    word_to_code_map = {k: i for i, k in enumerate(selected_words)}
    code_to_word_map = {v: k for k, v in word_to_code_map.items()}

    def process(self, data_map):
        wsst_df = data_map['word_start_stop_times']
        wsst_df['selected_word_code'] = wsst_df.word.map(self.word_to_code_map).fillna(-1)
        wsst_df['selected_word'] = np.where(wsst_df.word.isin(self.word_to_code_map), wsst_df.word, 'not_selected')

        key_cols = ['word', 'selected_word_code', 'stim_sentcode']
        vc_check = wsst_df[wsst_df.selected_word_code.ge(0)][key_cols].value_counts()
        assert vc_check.max() == 1, f"Unique values of {key_cols} should all be one - got : {vc_check.to_dict()}"

        return dict(word_start_stop_times=wsst_df)


class SelectWordsFromStartStopTime_Easy(DictTrf):
    selected_words = [  # 'HER',
        'TANK',
        'SUN',
        # 'SIZE',
        'FISH',
        'FENCE',
        # 'OVER',
        # 'BLUE',
        # 'DAYS',
        # 'LEFT',
        'BOY',
        # 'CLEAR',
        # 'YOUNG',
        # 'BEFORE',
        'GIRL',
        # 'GAVE'
    ]


@attr.s
@with_logger
class RandomStim(DictTrf):
    n = attr.ib()
    replace = attr.ib(False)
    index_source_stim_key = attr.ib('stim')
    output_key = attr.ib('random_stim')

    window_size = attr.ib(pd.Timedelta(0.5, 's'))
    slice_selection: Optional[slice] = attr.ib(None)

    def process(self, data_map):
        src_s = data_map[self.index_source_stim_key]

        random_stim = pd.Series(0, index=src_s.index)
        last_t = random_stim.index.max() - self.window_size

        # What we'll sample from to avoid sampling indices that cannot be used
        win_truncated_random_stim = random_stim.loc[:last_t]

        if self.slice_selection is not None:
            assert self.slice_selection.step is None

            if isinstance(self.slice_selection.start, (int, float)) and isinstance(self.slice_selection.stop, (int, float)):
                self.logger.info("Fractional slice detected - converting to fraction of the time stamps ")
                _max_t = win_truncated_random_stim.index.max()
                _slice = slice(_max_t * self.slice_selection.start, _max_t * self.slice_selection.stop)
            elif isinstance(self.slice_selection.start, str) and isinstance(self.slice_selection.stop, str):
                _slice = self.slice_selection
            else:
                raise ValueError(f"Don't understand slice: {self.slice_selection}")

            self.logger.info(
                f"Applying slice selection {_slice} to {len(win_truncated_random_stim)} samples in "
                f"{win_truncated_random_stim.index.min()} - {win_truncated_random_stim.index.max()}"
            )
            win_truncated_random_stim = win_truncated_random_stim.loc[_slice]
            self.logger.info(f"After slice selection, can sample from {len(win_truncated_random_stim)} samples")

        random_ix = win_truncated_random_stim.sample(n=self.n, replace=self.replace).index.unique()
        self.logger.info(f"Sampled {len(random_ix)} times from {self.index_source_stim_key}")
        random_stim.loc[random_ix] = 1
        self.logger.info(f"Total random indices in output: {random_stim.sum()}")

        return {self.output_key: random_stim}


def object_as_key_or_itself(key_or_value, remap=None):
    """
    Returns a value from (in order):
        - remap[key_or_value]
        - remap
        - key_or_value
    """
    if isinstance(remap, dict):
        value = remap[key_or_value]
    elif remap is not None:
        value = remap
    elif remap is None:
        value = key_or_value
    else:
        raise ValueError(f"Dont know how to handle remap of type: {type(remap)}")
    return value


@attr.s
@with_logger
class WindowSampleIndicesFromIndex(DictTrf):
    stim_key = attr.ib('stim')
    fs_key = attr.ib('fs_signal')
    index_shift = attr.ib(None)
    stim_target_value = attr.ib(1)
    method = attr.ib('target_equality')
    window_size = attr.ib(pd.Timedelta(0.5, 's'))
    sample_n = attr.ib(None)
    stim_value_remap = attr.ib(None)
    stim_pre_process_f = attr.ib(None)

    @classmethod
    def step_through_target_indexes(cls, stim, target_indexes, win_size, index_shift, expected_window_samples):
        valid_indices = list()
        for offs in target_indexes:
            _s = stim.loc[offs + index_shift:offs + win_size + index_shift]
            if len(_s) >= expected_window_samples:
               valid_indices.append(_s.iloc[:expected_window_samples].index)
            else:
                cls.logger.warning(f"Stim had {len(_s)} samples, not meeting expected size of {expected_window_samples}")

        return valid_indices

        #return [stim.loc[offs + index_shift:offs + win_size + index_shift].iloc[:expected_window_samples].index
        #        for offs in target_indexes
        #        if len(stim.loc[offs + index_shift:offs + win_size + index_shift]) >= expected_window_samples]

    def process(self, data_map):
        stim = data_map[self.stim_key]
        fs = data_map[self.fs_key]
        existing_sample_indices_map = data_map.get('sample_index_map')
        existing_indices_sources_map = data_map.get('index_source_map')

        existing_sample_indices_map = dict() if existing_sample_indices_map is None else existing_sample_indices_map
        existing_indices_sources_map = dict() if existing_indices_sources_map is None else existing_indices_sources_map
        sample_indices = dict()
        indices_sources = dict()

        stim_pre_process_f = self.stim_pre_process_f if self.stim_pre_process_f is not None else lambda _stim: _stim
        win_size = self.window_size

        index_shift = pd.Timedelta(0, 's') if self.index_shift is None else self.index_shift
        expected_window_samples = int(fs * win_size.total_seconds())

        if self.method == 'target_equality':
            target_indexes = (
                    stim.pipe(stim_pre_process_f) == self.stim_target_value).pipe(
                lambda s: s[s] if self.sample_n is None else (s[s].sample(n=self.sample_n) if len(s[s]) > self.sample_n else s[s])).index.tolist()
            # target_indices = [stim.loc[offs + index_shift:offs + win_size + index_shift].iloc[:expected_window_samples].index
            #                  for offs in target_indexes
            #                  if len(stim.loc[offs + index_shift:offs + win_size + index_shift]) >= expected_window_samples]
            target_indices = self.step_through_target_indexes(stim, target_indexes, win_size, index_shift,
                                                              expected_window_samples)

            stim_key = object_as_key_or_itself(self.stim_target_value, self.stim_value_remap)
            sample_indices[stim_key] = sample_indices.get(stim_key, list()) + target_indices
            indices_sources[stim_key] = indices_sources.get(stim_key, self.stim_key)
        elif self.method == 'unique_values':
            g = stim[stim >= 0].pipe(lambda s: s.groupby(s))
            # g = stim.pipe(lambda s: s.groupby(s))
            for gcode, gdf in g:
                if self.sample_n is None or self.sample_n >= len(gdf):
                    target_indexes = gdf.index.tolist()
                else:
                    # TODO / WARNGING: will evenly sample across any grouping of the stim value
                    #                   - e.g. same word, different sentence
                    target_indexes = gdf.sample(self.sample_n).index.tolist()

                target_indices = self.step_through_target_indexes(stim, target_indexes, win_size, index_shift,
                                                                  expected_window_samples)
                stim_key = object_as_key_or_itself(gcode, self.stim_value_remap)
                self.logger.info(
                    f"Unique Code {gcode} (key={stim_key}) has {len(target_indexes)} taken from {len(gdf)}"
                )
                sample_indices[gcode] = sample_indices.get(gcode, list()) + target_indices
                indices_sources[gcode] = indices_sources.get(gcode, stim_key)

        else:
            raise ValueError(f"Don't understand method = '{self.method}'")

        if existing_sample_indices_map is not None:
            existing_sample_indices_map.update(sample_indices)
            sample_indices = existing_sample_indices_map

        if existing_indices_sources_map is not None:
            existing_indices_sources_map.update(indices_sources)
            indices_sources = existing_indices_sources_map

        return dict(sample_index_map=sample_indices, n_samples_per_window=expected_window_samples,
                    index_source_map=indices_sources)


@attr.s
@with_logger
class WindowSampleIndicesFromStim(DictTrf):
    stim_key = attr.ib('stim')
    fs_key = attr.ib('fs_signal')
    window_size = attr.ib(pd.Timedelta(0.5, 's'))

    # One of rising or falling
    target_onset_reference = attr.ib('rising')
    target_offset_reference = attr.ib('falling')
    target_onset_shift = attr.ib(pd.Timedelta(-0.50, 's'))
    target_offset_shift = attr.ib(pd.Timedelta(0., 's'))

    sample_n = attr.ib(None)
    max_target_region_size = attr.ib(600)
    stim_value_remap = attr.ib(None)

    def process(self, data_map):

        stim, fs = data_map[self.stim_key], data_map[self.fs_key]
        existing_sample_indices_map = data_map.get('sample_index_map')
        existing_indices_sources_map = data_map.get('index_source_map')

        win_size = self.window_size

        existing_sample_indices_map = dict() if existing_sample_indices_map is None else existing_sample_indices_map
        existing_indices_sources_map = dict() if existing_indices_sources_map is None else existing_indices_sources_map

        expected_window_samples = int(fs * win_size.total_seconds())
        # label_region_sample_size = int(fs * label_region_size.total_seconds())
        self.logger.info((fs, win_size))
        self.logger.info("Samples per window: %d" % expected_window_samples)

        # Will map of codes to list of indices into the stim signal:
        # word_code->List[pd.Index, pd.Index, ...]
        sample_indices = dict()
        indices_sources = dict()

        # TODO: This will not work for constant stim value (i.e. True/False, 1/0)?
        # TODO: Need to review UCSD data and how to write something that will work for its regions
        s_grp = stim[stim > 0].pipe(lambda _s: _s.groupby(_s))
        for stim_value, g_s in tqdm(s_grp, desc=f"Processing stim regions from '{self.stim_key}'"):
            start_t = g_s.index.min()
            stop_t = g_s.index.max()

            if self.target_onset_reference == 'rising':
                target_start_t = start_t + self.target_onset_shift
            elif self.target_onset_reference == 'falling':
                target_start_t = stop_t + self.target_onset_shift
            else:
                raise ValueError(f"Dont understand {self.target_onset_reference}")

            if self.target_offset_reference == 'rising':
                target_stop_t = start_t + self.target_offset_shift
            elif self.target_offset_reference == 'falling':
                target_stop_t = stop_t + self.target_offset_shift
            else:
                raise ValueError(f"Dont understand {self.target_offset_reference}")

            # Get the window starting indices for each region of interest
            # Note on :-expected_window_samples
            #   - this removes the last windows worth since windows starting here would have out of label samples
            # Commented this out - use the offsets to handle this?
            # target_start_ixes = stim[target_start_t:target_stop_t].index.tolist()#[:-expected_window_samples]
            s_ix = stim[target_start_t:target_stop_t].index
            assert s_ix.is_unique, f"Index between {target_start_t} and {target_stop_t} is not unique!"
            target_start_ixes = s_ix.tolist()  # [:-expected_window_samples]

            if self.sample_n:
                if self.sample_n > len(target_start_ixes):
                    print(
                        f"Warning: tried to sample {self.sample_n}, but only {len(target_start_ixes)} start ixes present")
                    to_iter = target_start_ixes
                else:
                    to_iter = np.random.choice(target_start_ixes, self.sample_n, replace=False)
            elif self.max_target_region_size is not None:
                to_iter = target_start_ixes[:self.max_target_region_size]
            else:
                to_iter = target_start_ixes

            # Go through the labeled region indices and pull a window of data
            target_indices = [stim.loc[offs:offs + win_size].iloc[:expected_window_samples].index
                              for offs in to_iter  # target_start_ixes[:max_target_region_size]
                              if len(stim.loc[offs:offs + win_size]) >= expected_window_samples]

            stim_key = object_as_key_or_itself(stim_value, self.stim_value_remap)
            sample_indices[stim_key] = sample_indices.get(stim_key, list()) + target_indices
            indices_sources[stim_key] = indices_sources.get(stim_key, self.stim_key)

        # Go through all samples - make noise if sample size is off (or should throw error?)
        size_d = dict()
        for k, _s in sample_indices.items():
            size_d[k] = len(_s)
            # cls.logger.info(f"Extracted {len(_s)} from stim_value={stim_value}")
            for i, _ixs in enumerate(_s):
                if len(_ixs) != expected_window_samples:
                    self.logger.warning(f"[{k}][{i}] ({len(_ixs)}): {_ixs}")

        self.logger.info(f"Number of samples keys in sample index: {pd.Series(size_d).value_counts().to_dict()}")
        self.logger.info(f"Windows coded to (i.e. target value): {', '.join(map(str, sample_indices.keys()))}")

        # Debug code printing the unique lengths of each window for each word code
        # print({k : sorted(list(set(map(len, _s)))) for k, _s in sample_indices.items()})
        if existing_sample_indices_map is not None:
            existing_sample_indices_map.update(sample_indices)
            sample_indices = existing_sample_indices_map

        if existing_indices_sources_map is not None:
            existing_indices_sources_map.update(indices_sources)
            indices_sources = existing_indices_sources_map

        return dict(sample_index_map=sample_indices, n_samples_per_window=expected_window_samples,
                    index_source_map=indices_sources)

