import os
import attr
import socket
from os import path
from os import environ

import pandas as pd

from dataclasses import dataclass, field

from typing import List, Optional, Type, ClassVar

from mmz import utils
from brain2vec.preprocessing import steps as ps
from sklearn.pipeline import Pipeline

from brain2vec.datasets import BaseDataset, DatasetOptions
from brain2vec.datasets.base_aspen import BaseASPEN

with_logger = utils.with_logger(prefix_name=__name__)

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')

os.environ['WANDB_CONSOLE'] = 'off'

logger = utils.get_logger(__name__)


@attr.s
@with_logger
class HarvardSentences(BaseASPEN):
    """
    """

    env_key = 'HARVARDSENTENCES_DATASET'
    default_hvs_path = path.join(pkg_data_dir, 'HarvardSentences')
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 default_hvs_path))
    mat_d_keys = dict(
        signal='sEEG_signal',
        signal_fs='fs_signal',
        audio='audio',
        audio_fs='fs_audio',
        stimcode='stimcode',
        electrodes=None,
        wordcode=None,
    )

    all_patient_maps = dict(UCSD={
        4: [('UCSD', 4, 1, 1)],
        5: [('UCSD', 5, 1, 1)],
        10: [('UCSD', 10, 1, 1)],
        18: [('UCSD', 18, 1, 1)],
        19: [('UCSD', 19, 1, 1)],
        22: [('UCSD', 22, 1, 1)],
        28: [('UCSD', 28, 1, 1)],
    })

    def make_pipeline_map(self, default='audio_gate'):
        """
        Pipeline parameters sometimes depend on the configuration of the dataset class,
        so for now it is bound method (not classmethod or staticmethod).
        """
        parse_arr_steps = [
            ('parse_signal', ps.ParseTimeSeriesArrToFrame(self.mat_d_keys['signal'],
                                                                self.mat_d_keys['signal_fs'],
                                                                1200, output_key='signal')),
            ('parse_audio', ps.ParseTimeSeriesArrToFrame(self.mat_d_keys['audio'],
                                                               self.mat_d_keys['audio_fs'],
                                                               48000, reshape=-1)),
            ('parse_stim', ps.ParseTimeSeriesArrToFrame(self.mat_d_keys['stimcode'],
                                                              self.mat_d_keys['signal_fs'],
                                                              # TODO: Check the default rate here - 1024?
                                                              1200, reshape=-1, output_key='stim')),
            ('parse_sensor_ras', ps.ParseSensorRAS()),
            ('extract_mfc', ps.ExtractMFCC())
        ]

        parse_input_steps = [
            ('sensor_selection', ps.IdentifyGoodAndBadSensors(sensor_selection=self.sensor_columns)),
            # TODO: Wave2Vec2 standardizes like this
            #  - but should we keep this in to match or should we batch norm at the top?
            #('rescale_signal', pipeline.StandardNormSignal()),
            ('subsample', ps.SubsampleSignal()),
            ('sent_from_start_stop', ps.SentCodeFromStartStopWordTimes()),
            ('all_stim', ps.CreateAllStim()),

        ]

        audio_gate_steps = [
            ('Threshold', ps.PowerThreshold(speaking_window_samples=48000 // 16,
                                                  silence_window_samples=int(48000 * 1.5),
                                                  speaking_quantile_threshold=0.85,
                                                  # silence_threshold=0.001,
                                                  silence_quantile_threshold=0.05,
                                                  n_silence_windows=35000,
                                                  # silence_n_smallest=30000,
                                                  #stim_key='speaking_region_stim'
                                                  stim_key='speaking_region_stim_mask'
                                                  )),
#            ('speaking_indices', pipeline.WindowSampleIndicesFromStim('stim_pwrt',
#                                                                      target_onset_shift=pd.Timedelta(-.5, 's'),
#                                                                      # input are centers, and output is a window of
#                                                                      # .5 sec so to center it, move the point (
#                                                                      # center) back .25 secods so that extracted 0.5
#                                                                      # sec window saddles the original center
#                                                                      # target_offset_shift=pd.Timedelta(-0.25, 's')
#                                                                      target_offset_shift=pd.Timedelta(-0.5, 's'),
#                                                                      #max_target_region_size=300
#                                                                      sample_n=20000,
#                                                                      )),
            ('speaking_indices', ps.WindowSampleIndicesFromIndex('stim_pwrt',
                                                                      # Center the extracted 0.5 second window
                                                                      index_shift=pd.Timedelta(-0.25, 's'),
                                                                      stim_value_remap=1,
                                                                      sample_n=10000,
                                                                      )),
            ('silence_indices', ps.WindowSampleIndicesFromIndex('silence_stim_pwrt_s',
                                                                      # Center the extracted 0.5 second window
                                                                      index_shift=pd.Timedelta(-0.25, 's'),
                                                                      stim_value_remap=0,
                                                                      sample_n=10000,
                                                                      ))
        ]
        audio_gate_all_region_steps = [
                                          ('Threshold', ps.PowerThreshold(speaking_window_samples=48000 // 16,
                                                  silence_window_samples=int(48000 * 1.5),
                                                  speaking_quantile_threshold=0.85,
                                                  # silence_threshold=0.001,
                                                  silence_quantile_threshold=0.05,
                                                  n_silence_windows=35000,
                                                  # silence_n_smallest=30000,
                                                  #stim_key='speaking_region_stim'
                                                  stim_key='all_stim'
                                                  ))
        ] + audio_gate_steps[1:]

        start_stop_steps = [('new_mtss', ps.AppendExtraMultiTaskStartStop()),
                                                 # Stims from Start-stop-times
                                                 ('speaking_word_stim', ps.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='start_t',
                                                                        stop_t_column='stop_t',
                                                                        stim_output_name='speaking_word_stim',
                                                 )),
                                                 ('listening_word_stim', ps.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='listening_word_start_t',
                                                                        stop_t_column='listening_word_stop_t',
                                                                        stim_output_name='listening_word_stim',
                                                 )),
                                                 ('mouthing_word_stim', ps.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='mouthing_word_start_t',
                                                                        stop_t_column='mouthing_word_stop_t',
                                                                        stim_output_name='mouthing_word_stim',
                                                 )),
                                                ('imagining_word_stim', ps.NewStimFromRegionStartStopTimes(
                                                                        start_t_column='imagining_word_start_t',
                                                                        stop_t_column='imagining_word_stop_t',
                                                                        stim_output_name='imagining_word_stim',
                                                )),

                            ('speaking_region_stim', ps.NewStimFromRegionStartStopTimes(
                                start_t_column='speaking_region_start_t',
                                stop_t_column='speaking_region_stop_t',
                                stim_output_name='speaking_region_stim',
                            )),
                            ('listening_region_stim', ps.NewStimFromRegionStartStopTimes(
                                start_t_column='listening_region_start_t',
                                stop_t_column='listening_region_stop_t',
                                stim_output_name='listening_region_stim',
                            )),
                            ('mouthing_region_stim', ps.NewStimFromRegionStartStopTimes(
                                start_t_column='mouthing_region_start_t',
                                stop_t_column='mouthing_region_stop_t',
                                stim_output_name='mouthing_region_stim',
                            )),
                            ('imagining_region_stim', ps.NewStimFromRegionStartStopTimes(
                                start_t_column='imagining_region_start_t',
                                stop_t_column='imagining_region_stop_t',
                                stim_output_name='imagining_region_stim',
                            ))
                            ]

        region_kws = dict(
            target_onset_shift=pd.Timedelta(.5, 's'),
            target_offset_shift=pd.Timedelta(-1, 's'),
            sample_n=1000
        )
        region_from_word_kws = dict(
            target_onset_shift=pd.Timedelta(-.5, 's'),
            target_offset_shift=pd.Timedelta(-0.5, 's'),
        )
        select_words = ps.SelectWordsFromStartStopTimes()
        p_map = {
            'random_sample': Pipeline(parse_arr_steps + parse_input_steps
                                      + [('rnd_stim', ps.RandomStim(10_000)),
                                         ('rnd_indices', ps.WindowSampleIndicesFromIndex(stim_key='random_stim'))]
                                      + [('output', 'passthrough')]),

            'random_sample_pinknoise': Pipeline(parse_arr_steps + parse_input_steps +
                                                [
                                                    ('pinknoise', ps.ReplaceSignalWithPinkNoise()),
                                                    ('rnd_stim', ps.RandomStim(10_000)),
                                                    ('rnd_indices',
                                                     ps.WindowSampleIndicesFromIndex(stim_key='random_stim'))]
                                                + [('output', 'passthrough')]
                                                ),

            # -----
            # Directly from audio
            'audio_gate': Pipeline(parse_arr_steps + parse_input_steps  + start_stop_steps  #+ parse_stim_steps
                                   + audio_gate_steps + [('output', 'passthrough')]),
            'audio_gate_all_region': Pipeline(parse_arr_steps + parse_input_steps + start_stop_steps  # + parse_stim_steps
                                   + audio_gate_all_region_steps + [('output', 'passthrough')]),

            'region_classification': Pipeline(parse_arr_steps + parse_input_steps + start_stop_steps
                                                             + [
                                                                 # Indices from Stim - these populate the class labels
                                                                 ('speaking_indices',
                                                                  ps.WindowSampleIndicesFromStim(
                                                                      'speaking_region_stim',
                                                                      stim_value_remap=0, **region_kws)),
                                                                 ('listening_indices',
                                                                  ps.WindowSampleIndicesFromStim(
                                                                      'listening_region_stim',
                                                                      stim_value_remap=1, **region_kws)),
                                                                 ('mouthing_indices',
                                                                  ps.WindowSampleIndicesFromStim(
                                                                      'mouthing_region_stim',
                                                                      stim_value_remap=2, **region_kws)),
                                                                 ('imagining_indices',
                                                                  ps.WindowSampleIndicesFromStim(
                                                                      'imagining_region_stim',
                                                                      stim_value_remap=3, **region_kws)),
                                                                 ('output', 'passthrough')
                                                             ]),

            'region_classification_from_word_stim': Pipeline(parse_arr_steps + parse_input_steps + start_stop_steps
                                              + [
                                                # Indices from Stim - these populate the class labels
                                                ('speaking_indices', ps.WindowSampleIndicesFromStim(
                                                    'speaking_word_stim',
                                                    stim_value_remap=0,
                                                    **region_from_word_kws
                                                )),
                                                 ('listening_indices', ps.WindowSampleIndicesFromStim(
                                                    'listening_word_stim',
                                                    stim_value_remap=1,
                                                    **region_from_word_kws
                                                 )),
                                                 ('mouthing_indices', ps.WindowSampleIndicesFromStim(
                                                    'mouthing_word_stim',
                                                    stim_value_remap=2,
                                                    **region_from_word_kws

                                                 )),
                                                 ('imagining_indices', ps.WindowSampleIndicesFromStim(
                                                    'imagining_word_stim',
                                                    stim_value_remap=3,
                                                    **region_from_word_kws
                                                 )),('output', 'passthrough')]),

            'audio_gate_speaking_only': Pipeline(parse_arr_steps + parse_input_steps  + start_stop_steps
                                                 # Slice out the generation of the silence stim data - only speaking
                                                 + audio_gate_steps[:-1] + [('output', 'passthrough')]),

            'word_classification': Pipeline(parse_arr_steps + parse_input_steps  + start_stop_steps
                                                 # Slice out the generation of the silence stim data - only speaking
                                                 #+ audio_gate_steps +
                                            + [
                                                ('select_words_from_wsst', select_words),
                                                ('selected_speaking_word_stim', ps.NewStimFromRegionStartStopTimes(
                                                    start_t_column='start_t',
                                                    stop_t_column='stop_t',
                                                    label_column='selected_word',
                                                    code_column='selected_word_code',
                                                    stim_output_name='selected_speaking_word_stim',
                                                    default_stim_value=-1)),
                                                ('word_indices', ps.WindowSampleIndicesFromIndex(
                                                    'selected_speaking_word_stim',
                                                    method='unique_values',
                                                    stim_value_remap=select_words.code_to_word_map)),
                                                ##('word_indices', pipeline.WindowSampleIndicesFromStim(
                                                #    'selected_speaking_word_stim',
                                                #    target_onset_shift=pd.Timedelta(0, 's'),
                                                #    target_offset_shift=pd.Timedelta(0, 's'),
                                                #    stim_value_remap=select_words.code_to_word_map,
                                                #)),
                                                ('output', 'passthrough')]
                                            )

#            'audio_gate_imagine': Pipeline(parse_arr_steps + parse_input_steps + [
#                # Creates listening, imagine, mouth
#                #('multi_task_start_stop', pipeline.MultiTaskStartStop()),
#                # Creates the word_stim and sentence_stim from the start stop of imagine
#                ('stim_from_start_stop', pipeline.SentenceAndWordStimFromRegionStartStopTimes(start_t_column='imagine_start_t',
#                                                                                              stop_t_column='imagine_stop_t')),
#                # creat stim for listening (i.e. not speaking or active) that we'll use for silent
#                ('stim_from_listening', pipeline.SentenceAndWordStimFromRegionStartStopTimes(start_t_column='listening_region_start_t',
#                                                                                             stop_t_column='listening_region_stop_t',
#                                                                                             word_stim_output_name='listening_word_stim',
#                                                                                             sentence_stim_output_name='listening_sentence_stim',
#                                                                                             set_as_word_stim=False)),
#                # Target index extraction - word stim is the imagine stim extracted above
#                ('speaking_indices', pipeline.WindowSampleIndicesFromStim('word_stim',
#                                                                          target_onset_shift=pd.Timedelta(-.5, 's'),
#                                                                          target_offset_shift=pd.Timedelta(-0.5, 's'),
#                                                                          )),
#                # Negative target index extraction - use listening regions for negatives
#                ('silent_indices', pipeline.WindowSampleIndicesFromStim('listening_word_stim',
#                                                                        target_onset_shift=pd.Timedelta(.5, 's'),
#                                                                        target_offset_shift=pd.Timedelta(-0.5, 's'),
#                                                                        stim_value_remap=0,
#                                                                        )),
#
#                ('output', 'passthrough')
#            ]),

        }

        p_map['default'] = p_map[default]

        return p_map

    @classmethod
    def make_filename(cls, patient, session, trial, location):
        """
        UCSD04_Task_1.mat  UCSD10_Task_1.mat  UCSD19_Task_1.mat  UCSD28_Task_1.mat
        UCSD05_Task_1.mat  UCSD18_Task_1.mat  UCSD22_Task_1.mat
        """
        cls.logger.info("Harvard sentences only uses location and patient identifiers")
        loc_map = cls.all_patient_maps.get(location)
        if loc_map is None:
            raise KeyError(f"Valid locations: {list(cls.all_patient_maps.keys())}")

        fname = f"{location}{patient:02d}_Task_1.mat"

        return fname


BaseDataset.register_dataset('hvs', HarvardSentences)


@dataclass
class HarvardSentencesDatasetOptions(DatasetOptions):
    dataset_name: str = 'hvs'
    train_sets: str = 'UCSD-19'
    flatten_sensors_to_samples: bool = True
    extra_output_keys: Optional[str] = 'sensor_ras_coord_arr'


