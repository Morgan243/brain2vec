import os
import attr
import socket
from glob import glob
from os import path
from os import environ

import pandas as pd
import numpy as np
import scipy.io

import torch
import torchvision.transforms
from torch.utils import data as tdata

from tqdm.auto import tqdm
from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable

from typing import List, Optional, Type, ClassVar

from ecog_speech import feature_processing, utils, pipeline
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
class NorthwesternWords(BaseASPEN):
    """
    Northwestern-style data: one spoken word per cue, aligned brain and audio data

    This class can load multiple trails as once - ensuring correct windowing, but allowing
    for torch data sampling and other support.
    """

    env_key = 'NORTHWESTERNWORDS_DATASET'
    default_base_path = environ.get(env_key,
                                    path_map.get(socket.gethostname(),
                                                 path.join(pkg_data_dir,
                                                           'SingleWord')
                                                 ))
    mat_d_keys = dict(
        signal='ECOG_signal',
        signal_fs='fs_signal',
        audio='audio',
        audio_fs='fs_audio',
        stimcode='stimcode',
        electrodes='electrodes',
        wordcode='wordcode',
    )

    mc_patient_set_map = {
        19: [('MayoClinic', 19, 1, 1),
             ('MayoClinic', 19, 1, 2),
             ('MayoClinic', 19, 1, 3)],

        21: [('MayoClinic', 21, 1, 1),
             ('MayoClinic', 21, 1, 2)],

        22: [('MayoClinic', 22, 1, 1),
             ('MayoClinic', 22, 1, 2),
             ('MayoClinic', 22, 1, 3)],

        24: [('MayoClinic', 24, 1, 2),
             ('MayoClinic', 24, 1, 3),
             ('MayoClinic', 24, 1, 4)],

        # 25: [('MayoClinic', 25, 1, 1),
        #     ('MayoClinic', 25, 1, 2)],

        26: [('MayoClinic', 26, 1, 1),
             ('MayoClinic', 26, 1, 2)],
    }

    nw_patient_set_map = {
        1: [
            ('Northwestern', 1, 1, 1),
            ('Northwestern', 1, 1, 2),
            # ('Northwestern', 1, 1, 3),
        ],
        2: [
            ('Northwestern', 2, 1, 1),
            ('Northwestern', 2, 1, 2),
            ('Northwestern', 2, 1, 3),
            ('Northwestern', 2, 1, 4),
        ],
        3: [
            ('Northwestern', 3, 1, 1),
            ('Northwestern', 3, 1, 2),
        ],
        4: [
            ('Northwestern', 4, 1, 1),
            ('Northwestern', 4, 1, 2),
        ],
        5: [
            ('Northwestern', 5, 1, 2),
            ('Northwestern', 5, 1, 3),
            ('Northwestern', 5, 1, 4),
        ],
        6: [
            ('Northwestern', 6, 1, 7),
            ('Northwestern', 6, 1, 9),
        ],
    }

    syn_patient_set_map = {
        1: [('Synthetic', 1, 1, 1)],
        2: [('Synthetic', 2, 1, 1)],
        3: [('Synthetic', 3, 1, 1)],
        4: [('Synthetic', 4, 1, 1)],
        5: [('Synthetic', 5, 1, 1)],
        6: [('Synthetic', 6, 1, 1),
            ('Synthetic', 6, 1, 2)],
        7: [('Synthetic', 7, 1, 1),
            ('Synthetic', 7, 1, 2)],
        8: [('Synthetic', 8, 1, 1),
            ('Synthetic', 8, 1, 2)],
    }

    all_patient_maps = dict(MC=mc_patient_set_map,
                            SN=syn_patient_set_map,
                            NW=nw_patient_set_map)
    fname_prefix_map = {'MayoClinic': 'MC', 'Synthetic': 'SN', 'Northwestern': 'NW'}
    tuple_to_sets_str_map = {t: f"{l}-{p}-{i}"
                             for l, p_d in all_patient_maps.items()
                             for p, t_l in p_d.items()
                             for i, t in enumerate(t_l)}

    #######
    ## Path handling
    @classmethod
    def make_filename(cls, patient, session, trial, location):
        if location in cls.fname_prefix_map:  # == 'Mayo Clinic':
            return f"{cls.fname_prefix_map.get(location)}{str(patient).zfill(3)}-SW-S{session}-R{trial}.mat"
        else:
            raise ValueError("Don't know location " + location)


BaseDataset.register_dataset('nww', NorthwesternWords)


@dataclass
class NorthwesternWordsDatasetOptions(DatasetOptions):
    dataset_name: str = 'nww'
    train_sets: str = 'MC-21-0'
