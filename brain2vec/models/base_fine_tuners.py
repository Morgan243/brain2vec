import torch


class MultiChannelFromSingleChannel(torch.nn.Module):
    def __init__(self, input_shape, model_1d, model_dim=1, forward_kws=None, concat_axis=2,
                 model_output_key='x'):
        super().__init__()
        self.input_shape = input_shape
        self.model_1d = model_1d
        self.model_dim = model_dim
        self.forward_kws = dict(features_only=True, mask=False) if forward_kws is None else forward_kws
        self.concat_axis = concat_axis

        self.model_output_key = model_output_key

        self.T, self.S, self.C = self.model_1d.T, self.input_shape[0], self.model_1d.C

    def forward(self, input_d: dict):
        ras_arr = input_d['sensor_ras_coord_arr']
        x_arr = input_d['signal_arr']

        outputs_l = list()
        for i in range(x_arr.shape[self.model_dim]):
            _input_d = dict(sensor_ras_coord_arr=ras_arr.select(self.model_dim, i).unsqueeze(1),
                            signal_arr=x_arr.select(self.model_dim, i).unsqueeze(1))
            with torch.no_grad():
                outputs_l.append(self.model_1d(_input_d, **self.forward_kws)[self.model_output_key])

        output_arr = torch.cat([_x.unsqueeze(self.concat_axis) for _x in outputs_l], self.concat_axis)

        return dict(output=output_arr, target_arr=input_d['target_arr'])


class MultiChannelFineTuner(torch.nn.Module):
    """Takes a (usually pre trained) model that operates on single channels, and applies it to
    inputs with multiple channels"""

    def __init__(self, pre_trained_1d_model, output_model, pre_trained_model_forward_kws,
                 pre_trained_model_output_key, concat_axis=1):
        super().__init__()
        self.pre_trained_model = pre_trained_1d_model
        self.output_model = output_model
        self.pre_trained_model_output_key = pre_trained_model_output_key
        self.pre_trained_model_forward_kws = pre_trained_model_forward_kws

    def forward(self, input_d, features_only=False, mask=True):
        X = input_d['signal_arr']
        pass


class FineTuner(torch.nn.Module):
    def __init__(self, pre_trained_model, output_model, pre_trained_model_forward_kws,
                 pre_trained_model_output_key, auto_eval_mode=True, freeze_pre_train_weights=True):
        super().__init__()
        self.pre_trained_model = pre_trained_model
        self.output_model = output_model
        self.pre_trained_model_output_key = pre_trained_model_output_key
        self.pre_trained_model_forward_kws = pre_trained_model_forward_kws
        self.auto_eval_mode = auto_eval_mode
        self.freeze_pre_train_weights = freeze_pre_train_weights

    def forward(self, x):
        # Make sure train time processes are off - e.g. dropout, batchnorm stats collection
        if self.auto_eval_mode:
            self.pre_trained_model = self.pre_trained_model.eval()

        # Not really necessary, but just another layer of turning off grad
        if self.freeze_pre_train_weights:
            with torch.no_grad():
                pt_out = self.pre_trained_model(x,
                                                **(dict() if self.pre_trained_model_forward_kws is None
                                                   else self.pre_trained_model_forward_kws))
        else:
            pt_out = self.pre_trained_model(x,
                                            **(dict() if self.pre_trained_model_forward_kws is None
                                               else self.pre_trained_model_forward_kws))

        return self.output_model(pt_out[self.pre_trained_model_output_key])
