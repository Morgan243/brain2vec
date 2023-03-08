import os
from torch.utils import data as tdata
from dataclasses import dataclass, field
from simple_parsing.helpers import JsonSerializable

from typing import List, Optional, Type, ClassVar, Dict, Union

from mmz import utils

from mmz.datasets import *

with_logger = utils.with_logger(prefix_name=__name__)

path_map = dict()
pkg_data_dir = os.path.join(os.path.split(os.path.abspath(__file__))[0], '../data')

os.environ['WANDB_CONSOLE'] = 'off'

logger = utils.get_logger(__name__)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

@dataclass
class MultiSensorOptions:
    flatten_sensors_to_samples: bool = False
    """Sensors will bre broken up into sensors - inputs beceome (1, N-timesteps) samples (before batching)"""
    random_sensors_to_samples: bool = False


#########
# Torchvision style transformations for use in the datasets and loaders
@attr.s
class RollDimension:
    """
    Shift all values in a dimension by a random amount, values "rolled off"
    reappaer at the opposite side.
    """
    roll_dim = attr.ib(1)
    min_roll = attr.ib(-2)
    max_roll = attr.ib(2)
    return_roll_amount = attr.ib(False)

    def __call__(self, sample):
        roll_amount = int(np.random.random_integers(self.min_roll,
                                                    self.max_roll))
        # return torch.roll(sample, roll_amount, self.roll_dim)
        r_sample = np.roll(sample, roll_amount, self.roll_dim)

        if self.return_roll_amount:
            ret = r_sample, roll_amount
        else:
            ret = r_sample

        return ret


@attr.s
class ShuffleDimension:
    """
    Shuffle a dimension of the input sample
    """
    shuffle_dim = attr.ib(1)

    def __call__(self, sample):
        sample = np.copy(sample)
        # TODO/WARNING : probably won't work for more than 2d?
        # swap the shuffle_dim to the zeroth index
        sample = np.transpose(sample, [self.shuffle_dim, 0])
        # shuffle on the 0-dim in place
        np.random.shuffle(sample)
        # Swap the shuffle_dim back - i.e. do same transpose again
        sample = np.transpose(sample, [self.shuffle_dim, 0])
        return sample


class SelectFromDim:
    """Expects numpy arrays - use in Compose - models.base.Select can be used in Torch """

    def __init__(self, dim=0, index='random', keep_dim=True):
        super(SelectFromDim, self).__init__()
        self.dim = dim
        self.index = index
        self.keep_dim = keep_dim

    def __call__(self, x):
        ix = self.index
        if isinstance(self.index, str) and self.index == 'random':
            ix = np.random.randint(0, x.shape[self.dim])

        x = np.take(x, indices=ix, axis=self.dim)

        if self.keep_dim:
            x = np.expand_dims(x, self.dim)

        return x


@attr.s
class RandomIntLike:
    """
    Produce random integers in the specified range with the same shape
    as the input sample
    """
    low = attr.ib(0)
    high = attr.ib(2)

    def __call__(self, sample):
        return torch.randint(self.low, self.high, sample.shape, device=sample.device).type_as(sample)


@attr.s
class BaseDataset(tdata.Dataset):
    env_key = None
    fs_signal = attr.ib(None, init=False)

    _dataset_registry: ClassVar[Dict] = dict()

    @classmethod
    def make_dataloader(cls, dset, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      shuffle=False, pin_memory=False, **kwargs):
        if random_sample:
            if batches_per_epoch is None:
                # batches_per_epoch = len(dset) // batch_size
                batches_per_epoch = int(np.ceil(len(dset) / batch_size))

            dataloader = tdata.DataLoader(dset, batch_size=batch_size,
                                          sampler=tdata.RandomSampler(dset,
                                                                      replacement=True,
                                                                      num_samples=batches_per_epoch * batch_size),
                                          shuffle=shuffle, num_workers=num_workers,
                                          pin_memory=pin_memory,
                                          **kwargs)
        else:
            dataloader = tdata.DataLoader(dset, batch_size=batch_size,
                                          shuffle=shuffle, num_workers=num_workers,
                                          pin_memory=pin_memory,
                                          **kwargs)
        return dataloader

    def to_dataloader(self, batch_size=64, num_workers=2,
                      batches_per_epoch=None, random_sample=True,
                      shuffle=False, pin_memory=False, **kwargs):
        return self.make_dataloader(self, batch_size=batch_size, num_workers=num_workers,
                                    batches_per_epoch=batches_per_epoch, random_sample=random_sample,
                                    shuffle=shuffle, pin_memory=pin_memory, **kwargs)

    @classmethod
    def register_dataset(cls, dataset_name, dataset_cls):
        assert dataset_name not in cls._dataset_registry, f"{dataset_name} already in registry"
        cls._dataset_registry[dataset_name] = dataset_cls

    @classmethod
    def get_dataset_by_name(cls, dataset_name):
        try:
            dataset_cls = cls._dataset_registry[dataset_name]
        except KeyError as ke:
            print(f"Dataset {dataset_name} is not registered")
            raise

        return dataset_cls

    def get_feature_shape(self):
        raise NotImplementedError()

    def get_target_shape(self):
        raise NotImplementedError()


@dataclass
class DatasetOptions(JsonSerializable):
    dataset_name: str = None

    batch_size: int = 256
    batch_size_eval: Optional[int] = None
    batches_per_epoch: Optional[int] = None
    """If set, only does this many batches in an epoch - otherwise, will do enough batches to equal dataset size"""
    batches_per_eval_epoch: Optional[int] = None

    pre_processing_pipeline: str = 'default'
    #pipeline_params: Optional[Union[str, Dict]] = None
    pipeline_params: Optional[str] = None

    train_sets: str = None
    cv_sets: Optional[str] = None
    test_sets: Optional[str] = None

    sensor_columns: Optional[str] = None
    data_subset: str = 'Data'
    output_key: str = 'signal_arr'
    label_reindex_col: Optional[str] = None#"patient"

    extra_output_keys: Optional[str] = None
    random_sensors_to_samples: bool = False
    flatten_sensors_to_samples: bool = False
    split_cv_from_test: bool = True
    # power_q: float = 0.7
    random_targets: bool = False
    pin_memory: bool = False
    dl_prefetch_factor: Optional[int] = None

    n_dl_workers: int = 4
    n_dl_eval_workers: int = 6

    def make_eval_dl_kws(self):
        #self.dataset_map, self.dl_map, self.eval_dl_map
        eval_dl_kws = dict(num_workers=self.n_dl_eval_workers,
                           batch_size=self.batch_size if self.batch_size_eval is None else self.batch_size_eval,
                           batches_per_epoch=self.batches_per_eval_epoch,
                           shuffle=self.batches_per_eval_epoch is None,
                           pin_memory=self.pin_memory,
                           random_sample=self.batches_per_eval_epoch is not None)
        if self.dl_prefetch_factor is not None:
            eval_dl_kws['prefetch_factor'] = self.dl_prefetch_factor

        return eval_dl_kws

    def make_dl_kws(self):
        dl_kws = dict(num_workers=self.n_dl_workers, batch_size=self.batch_size,
                      batches_per_epoch=self.batches_per_epoch,
                      pin_memory=self.pin_memory,
                      shuffle=False, random_sample=True)
        if self.dl_prefetch_factor is not None:
            dl_kws['prefetch_factor'] = self.dl_prefetch_factor
        return dl_kws

    def make_datasets_and_loaders(self, dataset_cls=None, base_data_kws=None,
                                  train_data_kws=None, cv_data_kws=None, test_data_kws=None,
                                  train_sets_str=None, cv_sets_str=None, test_sets_str=None,
                                  train_p_tuples=None, cv_p_tuples=None, test_p_tuples=None,
                                  train_sensor_columns=None,
                                  pre_processing_pipeline=None,
                                  additional_transforms=None,
                                  train_split_kws=None, test_split_kws=None,
                                  #split_cv_from_test=True
                                  # additional_train_transforms=None, additional_eval_transforms=None,
                                  #num_dl_workers=None
                                  ) -> tuple:
        """
        Helper method to create instances of dataset_cls as specified in the command-line options and
        additional keyword args.
        Parameters
        ----------
        options: object
            Options object build using the utils module
        dataset_cls: Derived class of BaseDataset (default=None)
            E.g. NorthwesterWords
        train_data_kws: dict (default=None)
            keyword args to train version of the dataset
        cv_data_kws: dict (default=None)
            keyword args to cv version of the dataset
        test_data_kws: dict (default=None)
            keyword args to test version of the dataset
        num_dl_workers: int (default=8)
            Number of workers in each dataloader. Can be I/O bound, so sometimes okay to over-provision

        Returns
        -------
        dataset_map, dataloader_map, eval_dataloader_map
            three-tuple of (1) map to original dataset (2) map to the constructed dataloaders and
            (3) Similar to two, but not shuffled and larger batch size (for evaluation)
        """
        base_data_kws = dict() if base_data_kws is None else base_data_kws
        if dataset_cls is None:
            dataset_cls = BaseDataset.get_dataset_by_name(self.dataset_name)

        if train_p_tuples is None:
            train_p_tuples = dataset_cls.make_tuples_from_sets_str(self.train_sets if train_sets_str is None
                                                                   else train_sets_str)
        if cv_p_tuples is None:
            cv_p_tuples = dataset_cls.make_tuples_from_sets_str(self.cv_sets if cv_sets_str is None
                                                                else cv_sets_str)
        if test_p_tuples is None:
            test_p_tuples = dataset_cls.make_tuples_from_sets_str(self.test_sets if test_sets_str is None
                                                                  else test_sets_str)

        if train_sensor_columns is None and self.sensor_columns is not None:
            train_sensor_columns = self.sensor_columns

        train_split_kws = dict() if train_split_kws is None else train_split_kws
        #test_split_kws = dict() if test_split_kws is None else test_split_kws

        logger.info("Train tuples: " + str(train_p_tuples))
        logger.info("CV tuples: " + str(cv_p_tuples))
        logger.info("Test tuples: " + str(test_p_tuples))

        if isinstance(self.pipeline_params, str):
            self.pipeline_params = eval(self.pipeline_params)

        base_kws = dict(pre_processing_pipeline=self.pre_processing_pipeline if pre_processing_pipeline is None
                                                else pre_processing_pipeline,
                        pipeline_params=self.pipeline_params,
                        data_subset=self.data_subset,
                        label_reindex_col=self.label_reindex_col,
                        extra_output_keys=self.extra_output_keys.split(',') if self.extra_output_keys is not None
                                          else None,
                        flatten_sensors_to_samples=self.flatten_sensors_to_samples)

        base_kws.update(base_data_kws)
        logger.info(f"Dataset base keyword arguments: {base_kws}")
        train_kws = dict(patient_tuples=train_p_tuples, **base_kws)
        cv_kws = dict(patient_tuples=cv_p_tuples, **base_kws)
        test_kws = dict(patient_tuples=test_p_tuples, **base_kws)

        if train_data_kws is not None:
            train_kws.update(train_data_kws)
        if cv_data_kws is not None:
            cv_kws.update(cv_data_kws)
        if test_data_kws is not None:
            test_kws.update(test_data_kws)

        dl_kws = self.make_dl_kws()
        logger.info(f"dataloader Keyword arguments: {dl_kws}")

        eval_dl_kws = self.make_eval_dl_kws()
        dataset_map = dict()
        logger.info("Using dataset class: %s" % str(dataset_cls))

        # Setup train dataset - there is always a train dataset
        train_dataset = dataset_cls(sensor_columns=train_sensor_columns, **train_kws)

        # Check for some special options on this DatasetOptions
        roll_channels = getattr(self, 'roll_channels', False)
        shuffle_channels = getattr(self, 'shuffle_channels', False)

        if roll_channels and shuffle_channels:
            raise ValueError("--roll-channels and --shuffle-channels are mutually exclusive")
        elif roll_channels:
            logger.info("-->Rolling channels transform<--")
            train_dataset.append_transform(
                RollDimension(roll_dim=0, min_roll=0,
                              max_roll=train_dataset.sensor_count - 1)
            )
        elif shuffle_channels:
            logger.info("-->Shuffle channels transform<--")
            train_dataset.append_transform(
                ShuffleDimension()
            )

        # Holdout should also use good for pt if that was set, otherwise, what training selected
        if train_sensor_columns == 'good_for_participant':
            holdout_sensor_columns = train_sensor_columns
        else:
            holdout_sensor_columns = train_dataset.selected_columns

        dataset_map['train'] = train_dataset

        # Check for explicit specification of patient tuples for a CV set
        if cv_kws['patient_tuples'] is not None:
            logger.info("+" * 50)
            logger.info(f"Using {cv_kws['patient_tuples']}")
            dataset_map['cv'] = dataset_cls(sensor_columns=holdout_sensor_columns, **cv_kws)
        # HVS is special case: CV set is automatic, and split at the participant-sentence code level
       #elif dataset_cls == HarvardSentences:
        elif dataset_cls.__name__ == 'HarvardSentences':
            logger.info("*" * 30)
            logger.info("Splitting on random key levels for harvard sentences (UCSD)")
            logger.info("*" * 30)
            _train, _test = train_dataset.split_select_random_key_levels(**train_split_kws)
            if test_split_kws is not None and self.split_cv_from_test:
                logger.info("Splitting out cv from test set")
                _cv, _test = _test.split_select_random_key_levels(**test_split_kws)
                dataset_map.update(dict(train=_train, cv=_cv, test=_test))
            elif test_split_kws is not None and not self.split_cv_from_test:
                logger.info("Splitting out cv from train set")
                _train, _cv = _train.split_select_random_key_levels(**test_split_kws)
                dataset_map.update(dict(train=_train, cv=_cv, test=_test))
            else:
                logger.info("Using cv from train set")
                dataset_map.update(dict(train=_train, cv=_test))

        # Otherwise, brute force split the window samples using size of data and dataset.select()
        else:
            logger.info("~" * 30)
            logger.info("Performing naive split at window level - expected for NWW datasets")
            logger.info("~" * 30)
            from sklearn.model_selection import train_test_split
            train_ixs, cv_ixes = train_test_split(range(len(train_dataset)))
            cv_nww = dataset_cls(data_from=train_dataset, **cv_kws).select(cv_ixes)
            train_dataset.select(train_ixs)
            dataset_map.update(dict(train=train_dataset,
                                    cv=cv_nww))

        if getattr(self, 'random_targets', False):
            logger.info("-->Randomizing target labels<--")
            class_val_to_label_d, class_labels = dataset_map['train'].get_target_labels()
            logger.info(f"Will use random number between 0 and {len(class_labels)}")
            dataset_map['train'].append_target_transform(
                RandomIntLike(low=0, high=len(class_labels))
            )

        # Test data is not required, but must be loaded with same selection of sensors as train data
        # TODO / Note: this could have a complex interplay if used tih flatten sensors or 2d data
        if test_kws['patient_tuples'] is not None:
            logger.info(f"Loading test set using KWS: {test_kws}")
            dataset_map['test'] = dataset_cls(sensor_columns=holdout_sensor_columns, **test_kws)
        else:
            logger.info(" - No test datasets provided - ")

        # dataset_map = dict(train=train_nww, cv=cv_nww, test=test_nww)
        if self.random_sensors_to_samples:
            additional_transforms = (list() if additional_transforms is None else additional_transforms)
            additional_transforms += [SelectFromDim(dim=0,
                                                    index='random',
                                                    keep_dim=True)]

        if isinstance(additional_transforms, list):
            dataset_map = {k: v.append_transform(additional_transforms)
                           for k, v in dataset_map.items()}

        dataloader_map = {k: v.to_dataloader(**dl_kws)
                          for k, v in dataset_map.items()}
        eval_dataloader_map = {k: v.to_dataloader(**eval_dl_kws)
                               for k, v in dataset_map.items()}

        return dataset_map, dataloader_map, eval_dataloader_map

