import pandas as pd
import torch

from mmz import utils, datasets

with_logger = utils.with_logger(prefix_name=__name__)

from mmz.models import *

from typing import Optional, Dict, List, Tuple


def make_model(options=None, nww=None, model_name=None, model_kws=None, print_details=True):
    from brain2vec.models import brain2vec

    if model_name == 'cog2vec' or model_name == 'brain2vec':
        m = brain2vec.Brain2Vec(**model_kws)
        m_kws = model_kws
    else:
        raise ValueError(f"Don't know how to load {model_name}")

    return m, m_kws


@with_logger
@attr.attrs
class Trainer:
    model_map = attr.ib()
    opt_map = attr.ib()

    train_data_gen = attr.ib()
    #input_key = attr.ib('ecog_arr')
    input_key = attr.ib('signal_arr')
    target_key = attr.ib('text_arr')

    #optim_kwargs = attr.ib(dict(weight_decay=0.2, lr=0.001))
    learning_rate = attr.ib(0.001)
    beta1 = attr.ib(0.5)

    criterion = attr.ib(torch.nn.BCELoss())
    extra_criteria = attr.ib(None) # regularizers here

    # How many epochs of not beating best CV loss by threshold before early stopping
    early_stopping_patience = attr.ib(None)
    # how much better the new score has to be (0 means at least equal)
    early_stopping_threshold = attr.ib(0)

    cv_data_gen = attr.ib(None)
    epochs_trained = attr.ib(0)
    device = attr.ib(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    cache_dataloaders = attr.ib(True)
    compile_models = attr.ib(True)

    weights_init_f = attr.ib(None)

    epoch_cb_history = attr.ib(attr.Factory(list), init=False)
    batch_cb_history = attr.ib(attr.Factory(list), init=False)
    model_regularizer = attr.ib(None)
    weight_decay = attr.ib(0)

    lr_adjust_on_cv_loss = attr.ib(False)
    lr_adjust_on_plateau_kws = attr.ib(None)
    #lr_adjust_metric = attr.ib('')
    model_name_to_lr_adjust = attr.ib(None)

    default_optim_cls = torch.optim.Adam

    @classmethod
    def set_default_optim(cls, optim):
        cls.default_optim_cls = optim
        return cls

    def __attrs_post_init__(self):
        if self.cache_dataloaders:
            self.logger.info("Caching with torch data in_memory_cach() unlimited memory")
            from torchdata.datapipes.iter import IterableWrapper
            self.train_data_gen = IterableWrapper(self.train_data_gen).in_memory_cache()
            if self.cv_data_gen is not None:
                self.cv_data_gen = IterableWrapper(self.cv_data_gen).in_memory_cache()

        self.model_map = {k: v.to(self.device) for k, v in self.model_map.items()}
        if self.compile_models and hasattr(torch, 'compile'):
            #torch._dynamo.config.suppress_errors = True
            #torch._dynamo.config.verbose = True
            self.model_map = {k: torch.compile(m) for k, m in self.model_map.items()}


        self.scheduler_map = dict()
        # Go throught the provided models to init, init their optimizers with their parameters, and setup their scheduler
        for k, m in self.model_map.items():
            # Init model weights
            if self.weights_init_f is not None:
                m.apply(self.weights_init_f)

            # Hard coded support for a some optimizers and their different constructions
            if k not in self.opt_map:
                if self.default_optim_cls == torch.optim.Adam:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                       lr=self.learning_rate,
                                                             weight_decay=self.weight_decay,
                                                             #weight_decay=0.9,
                                                       betas=(self.beta1, 0.999))
                elif self.default_optim_cls == torch.optim.RMSprop:
                    self.opt_map[k] = self.default_optim_cls(m.parameters(),
                                                             weight_decay=self.weight_decay,
                                                             lr=self.learning_rate)

            # Turn on LR scheduler and (no specific model to schedule or this is a specific model to adjust)
            if self.lr_adjust_on_plateau_kws and (self.model_name_to_lr_adjust is None
                                                  or k in self.model_name_to_lr_adjust):
                self.scheduler_map[k] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt_map[k],
                                                                                   verbose=True,
                                                                                   **self.lr_adjust_on_plateau_kws)

    def copy_model_states(self, models_to_exclude: Optional[List[str]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        models_to_exclude = list() if models_to_exclude is None else models_to_exclude
        return {k: copy_model_state(m) for k, m in self.model_map.items() if k not in models_to_exclude}

    def get_best_state(self, model_key='model'):
        if getattr(self, 'best_model_state', None) is not None:
            return self.best_model_state
        else:
            return copy_model_state(self.model_map[model_key])

    def train(self, n_epochs,
              epoch_callbacks=None,
              batch_callbacks=None,
              batch_cb_delta=3):

        epoch_callbacks = dict() if epoch_callbacks is None else epoch_callbacks
        batch_callbacks = dict() if batch_callbacks is None else batch_callbacks

        self.epoch_batch_res_map = dict()
        self.epoch_res_map = dict()

        self.epoch_cb_history += [{k: cb(self) for k, cb in epoch_callbacks.items()}]
        self.batch_cb_history = {k: list() for k in batch_callbacks.keys()}

        self.n_samples = len(self.train_data_gen)

        with tqdm(total=n_epochs,
                  desc='Training epoch',
                  dynamic_ncols=True
                  #ncols='100%'
                  ) as epoch_pbar:
            for epoch in range(self.epochs_trained, self.epochs_trained + n_epochs):
                epoch_batch_results = dict()
                with tqdm(total=self.n_samples, desc='-loss-', dynamic_ncols=True) as batch_pbar:
                    for i, data in enumerate(self.train_data_gen):
                        update_d = self.train_inner_step(epoch, data)

                        prog_msgs = list()
                        for k, v in update_d.items():
                            if k == 'n': continue

                            # TODO: What about spruious results - can't quite infer epoch?
                            #  Maybe do list of dicts instead?
                            # if this result key hasn't been seen, init a list of values
                            if k not in epoch_batch_results:
                                epoch_batch_results[k] = [v]
                            # eLse, just append the new value
                            else:
                                epoch_batch_results[k].append(v)
                            # Expecting numerics due to this code - take mean of the metric/loss
                            v_l = np.mean(epoch_batch_results[k])
                            # Just assume they are all the same size
                            #v_l = np.sum(epoch_batch_results[k]) / (self.n_samples * update_d['n'])
                            # build up the prog bar description string
                            prog_msgs.append(f"{k[:5]}: {np.round(v_l, 6)}")

                        msg = " || ".join(prog_msgs)
                        batch_pbar.set_description(msg)
                        batch_pbar.update(1)
                        if batch_cb_delta is None or (not i % batch_cb_delta):
                            for k, cb in batch_callbacks.items():
                                self.batch_cb_history[k].append(cb(self))

                self.epoch_batch_res_map[epoch] = epoch_batch_results
                self.epoch_res_map[epoch] = {k: np.mean(v) for k, v in epoch_batch_results.items()}

                self.epochs_trained += 1
                self.epoch_cb_history.append({k: cb(self) for k, cb in epoch_callbacks.items()})

                # Produce eval results if a cv dataloader was given
                if self.cv_data_gen:
                    cv_losses_d = self._eval(epoch, self.cv_data_gen)
                    #cv_l_mean = np.mean(cv_losses)
                    cv_l_mean = cv_losses_d['primary_loss']
                    self.epoch_res_map[epoch]['cv_loss'] = cv_l_mean
                    self.epoch_res_map[epoch]['cv_loss_d'] = cv_losses_d

                    for m_name, m_sched in self.scheduler_map.items():
                        m_sched.step(cv_l_mean)

                    if self.early_stopping_patience is not None:
                        self.last_best_cv_l = getattr(self, 'last_best_cv_l', np.inf)
                        if (self.last_best_cv_l - cv_l_mean) > self.early_stopping_threshold:
                            self.logger.info("-------------------------")
                            self.logger.info("---New best for early stopping---")
                            self.logger.info("-------------------------")
                            self.last_best_epoch = epoch
                            self.last_best_cv_l = cv_l_mean
                        elif (epoch - self.last_best_epoch) > self.early_stopping_patience:
                            self.logger.info("--------EARLY STOPPING----------")
                            self.logger.info(f"{epoch} - {self.last_best_epoch} > {self.early_stopping_patience} :: {cv_l_mean}, {self.last_best_cv_l}")
                            self.logger.info("-------------------------")
                            break

                epoch_pbar.update(1)
        return self.epoch_res_map

    # Reuses trainer.generate_outputs(), but not being used
    def _eval_v2(self, epoch_i, dataloader, model_key='model'):
        output_map = self.generate_outputs(model_key, CV=dataloader)['CV']
        mean_loss = np.mean(output_map['loss'])
        self.best_cv = getattr(self, 'best_cv', np.inf)
        new_best = mean_loss < self.best_cv
        if new_best:
            self.best_model_state = copy_model_state(self.model_map[model_key])
            self.best_model_epoch = epoch_i
            self.best_cv = mean_loss

        return output_map, new_best

    def _eval(self, epoch_i, dataloader, model_key='model'):
        """
        trainer's internal method for evaluating losses,
        snapshotting best models and printing results to screen
        """
        model = self.model_map[model_key].eval()
        self.best_cv = getattr(self, 'best_cv', np.inf)

        preds_l, actuals_l, loss_l = list(), list(), list()
        with torch.no_grad():
            with tqdm(total=len(dataloader), desc="Eval") as pbar:
                for i, _x in enumerate(dataloader):
                    preds = model(_x[self.input_key].to(self.device))
                    actuals = _x[self.target_key].to(self.device)
                    loss = self.criterion(preds, actuals)

                    loss_l.append(loss.detach().cpu().item())

                    pbar.update(1)

                mean_loss = np.mean(loss_l)
                desc = "Mean Eval Loss: %.5f" % mean_loss
                reg_l = 0.
                if self.model_regularizer is not None:
                    reg_l = self.model_regularizer(model)
                    desc += (" (+ %.6f reg loss = %.6f)" % (reg_l, mean_loss + reg_l))

                overall_loss = (mean_loss + reg_l)

                if overall_loss < self.best_cv:

                    self.best_model_state = copy_model_state(model)
                    self.best_model_epoch = epoch_i
                    self.best_cv = overall_loss
                    desc += "[[NEW BEST]]"

                pbar.set_description(desc)

        self.model_map['model'].train()
        return dict(primary_loss=overall_loss, cv_losses=loss_l)

        #return dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
        #            actuals=torch.cat(actuals_l).detach().cpu().int().numpy())

    def train_inner_step(self, epoch_i, data_batch):
        """
        Core training method - gradient descent - provided the epoch number and a batch of data and
        must return a dictionary of losses.
        """
        res_d = dict()

        model = self.model_map['model'].to(self.device)
        optim = self.opt_map['model']
        model = model.train()

        model.zero_grad()
        optim.zero_grad()

        input_arr = data_batch[self.input_key].to(self.device)
        actual_arr = data_batch[self.target_key].to(self.device)
        m_output = model(input_arr)

        crit_loss = self.criterion(m_output, actual_arr)
        res_d['crit_loss'] = crit_loss.detach().cpu().item()

        if self.model_regularizer is not None:
            reg_l = self.model_regularizer(model)
            res_d['bwreg'] = reg_l.detach().cpu().item()
        else:
            reg_l = 0

        loss = crit_loss + reg_l
        res_d['total_loss'] = loss.detach().cpu().item()
        loss.backward()
        optim.step()
        model = model.eval()
        return res_d

    def generate_outputs(self, model_key='model', **dl_map):
        """
        Evaluate a model the trainer has on a dictionary of dataloaders.
        """
        model = self.model_map[model_key].eval()
        return self.generate_outputs_from_model(model, dl_map, criterion=self.criterion, device=self.device,
                                                to_frames=False, input_key=self.input_key, target_key=self.target_key)

    @classmethod
    def generate_outputs_from_model_inner_step(cls, model, data_batch, criterion=None,
                                               input_key='ecog_arr', target_key='text_arr',
                                               device=None):
        _x_in = data_batch[input_key]
        _y = data_batch[target_key]
        if device is not None:
            _x_in = _x_in.to(device)
            _y = _y.to(device)

        preds = model(_x_in)
        ret = dict(preds=preds, actuals=_y)
        if criterion is not None:
            ret['criterion'] = criterion(preds, _y)

        return ret


    def generate_outputs_from_model(self, model, dl_map, criterion=None, device=None,
                                    to_frames=True, win_step=None, win_size=None,
                                    input_key='ecog_arr', target_key='text_arr') -> dict:
        """
        Produce predictions and targets for a mapping of dataloaders. B/c the trainer
        must know how to pair predictions and targets to train, this is implemented here.
        If model is in training mode, model is returned in training mode

        model: torch model
        dl_map: dictionary of dataloaders to eval on

        returns:
        output_map[dl_map_key][{"preds", "actuals"}]
        """
        model_in_training = model.training
        if device:
            model = model.to(device)

        model.eval()
        output_map = dict()
        with torch.no_grad():
            for dname, dl in dl_map.items():
                preds_l, actuals_l, criterion_l = list(), list(), list()
                #dset = dl.dataset

                #if hasattr(dset, 'data_maps'):
                #    assert len(dset.data_maps) == 1

                #data_map = next(iter(dset.data_maps.values()))
                res_d = dict()
                for _x in tqdm(dl, desc="Eval on [%s]" % str(dname)):
                    _inner_d = self.generate_outputs_from_model_inner_step(model, _x, input_key=input_key,
                                                                          target_key=target_key, device=device)

                    for k, v in _inner_d.items():
                        curr_v = res_d.get(k, list())
                        new_v = (curr_v + v) if isinstance(v, list) else (curr_v + [v])
                        res_d[k] = new_v

                output_map[dname] = {k: torch.cat(v_l).detach().cpu().numpy() #if len(v_l[0].shape) > 0
                                     #else torch.tensor(v_l).detach().cpu().numpy()
                                     for k, v_l in res_d.items()}
                #output_map[dname] = dict(preds=torch.cat(preds_l).detach().cpu().numpy(),
                #                         actuals=torch.cat(actuals_l).detach().cpu().int().numpy())
                                         #loss=torch.cat(criterion_l).detach().cpu().numpy())
                if to_frames:
                    t_ix = None
                    #if win_step is not None and win_size is not None:
                    #    t_ix = data_map['ecog'].iloc[range(win_size, data_map['ecog'].shape[0], win_step)].index
                    out_df = pd.DataFrame({k: v.squeeze() for k, v in output_map[dname].items()}, index=t_ix)
                    output_map[dname] = out_df
                    #output_map = {out_k: pd.DataFrame({k: v.squeeze() for k, v in preds_map.items()}, index=t_ix)
                    #              for out_k, preds_map in output_map.items()}

                if criterion is not None:
                    output_map[dname]['loss'] = torch.Tensor(criterion_l).detach().cpu().numpy()

        if model_in_training:
            model.train()

#        if to_frames:
#            t_ix = None
#            if win_step is not None and win_size is not None and data_map is not None:
#                t_ix = data_map['ecog'].iloc[range(win_size, data_map['ecog'].shape[0], win_step)].index
#
#            output_map = {out_k: pd.DataFrame({k: v.squeeze() for k, v in preds_map.items()}, index=t_ix)
#                          for out_k, preds_map in output_map.items()}

        return output_map


 ########
# Model Options
@dataclass
class ModelOptions(JsonSerializable, utils.SetParamsMixIn):
    non_hyperparams: ClassVar[Optional[list]] = ['device']

    @classmethod
    def get_all_model_hyperparam_names(cls):
        return [k for k, v in cls.__annotations__.items()
                if k not in cls.non_hyperparams]

    def make_model_kws(self, dataset=None, **kws):
        non_model_params = self.get_all_model_hyperparam_names()
        return {k: v for k, v in self.__annotations__.items() if k not in non_model_params}

    def make_model(self, dataset: Optional[datasets.BaseDataset] = None, in_channels=None, window_size=None):
        raise NotImplementedError()

    def make_model_regularizer_function(self, model):
        return None


@dataclass
class DNNModelOptions(ModelOptions):
    activation_class: str = 'PReLU'
    dropout: float = 0.
    dropout_2d: bool = False
    batch_norm: bool = False
    print_details: bool = True


@dataclass
class CNNModelOptions(DNNModelOptions):
    dense_width: Optional[int] = None
    n_cnn_filters: Optional[int] = None
    in_channel_dropout_rate: float = 0.

    def make_model_kws(self, dataset: Optional[Type[datasets.BaseDataset]] = None,
                       in_channels=None, window_size=None):
        return dict(in_channels=int(dataset.get_feature_shape()[0]) if in_channels is None else in_channels,
                    window_size=int(dataset.get_feature_shape()[-1]) if window_size is None else window_size,
                    dropout=self.dropout,
                    in_channel_dropout_rate=self.in_channel_dropout_rate,
                    dropout2d=self.dropout_2d,
                    batch_norm=self.batch_norm,
                    dense_width=self.dense_width,
                    n_cnn_filters=self.n_cnn_filters,
                    activation_cls=self.activation_class,
                    print_details=self.print_details)

    def make_model(self, dataset: Optional[Type[datasets.BaseDataset]] = None,
                   in_channels=None, window_size=None):
        model_kws = self.make_model_kws(dataset, in_channels=in_channels, window_size=window_size)
        return BaseCNN(**model_kws), model_kws


@dataclass
class BaseCNNModelOptions(CNNModelOptions):
    model_name: str = "base_cnn"
