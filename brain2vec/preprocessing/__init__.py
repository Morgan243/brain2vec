import numpy as np
import pandas as pd
from scipy import signal as sig


def peak_normalization(arr_s, output_type=int):
    """
    Divide by max value present in array, multiply by the datatypes max, convert to integer
    :param arr_s:
    :return:
    """
    return ((arr_s / arr_s.max()) * np.iinfo(arr_s.dtype).max).round().astype(output_type)


def compute_speaking_mask(wav, power_win_size=1024,
                          avg_power_thresh=50, avg_power_win_size=4096):
    """
    Uses rolling power estimate to gate the input - 1 or True is passthrough
    """
    pwr_s = wav.abs().rolling(power_win_size).mean()
    return (pwr_s > avg_power_thresh).rolling(avg_power_win_size).max() > 0


def speech_region_index_from_speech_mask(spk_mask_s):
    leading_edge_s = ((spk_mask_s.diff().replace(0, np.nan) + 1) / 2)
    word_trail_label_s = leading_edge_s.fillna(0).cumsum()
    word_index = spk_mask_s * word_trail_label_s

    # Incrementing negatives during silence, 0 everywhere else
    neg_neutral_cnt = -(word_trail_label_s - word_index - spk_mask_s) - 1

    # combine to make incrementing negatives identify leading silence regions
    # and incrementing positive labels for speech
    r_ix = pd.Series(np.where(neg_neutral_cnt == 0, word_index, neg_neutral_cnt),
                     index=word_index.index)
    return r_ix


def compute_speech_index(wav, speaking_mask_f=compute_speaking_mask):
    """
    Provide a function that
    """
    spk_mask_s = speaking_mask_f(wav).astype(int)
    return speech_region_index_from_speech_mask(spk_mask_s)


######
def fft(data, fs=1000):
    fft_vals = np.absolute(np.fft.rfft(data, axis=0))

    # Get frequencies for amplitudes in Hz
    fft_freq = np.fft.rfftfreq(len(data), 1.0 / fs)
    #return fft_freq, fft_vals
    s = pd.Series(fft_vals, index=fft_freq, name='fft_values')
    s.index.name = 'hz'
    return s


def filter(s, band, sfreq=1000, verbose=False, n_jobs=4,
           method='fir'):
    try:
        import mne
    except ImportError:
        print("Need MNE package to filter")
        raise
    if isinstance(s, pd.Series):
        _s = s.values
    elif isinstance(s, pd.DataFrame):
        _s = s.values.T
    else:
        _s = s

    _s = _s.astype('float64')

    filtered_phase_arr = mne.filter.filter_data(_s, sfreq,
                                                *band,
                                                verbose=verbose,
                                                method=method,
                                                n_jobs=n_jobs)

    if isinstance(s, pd.Series):
        ret = pd.Series(filtered_phase_arr, index=s.index, name=s.name)

    elif isinstance(s, pd.DataFrame):
        ret = pd.DataFrame(filtered_phase_arr.T,
                           columns=s.columns,
                           index=s.index)
    else:
        ret = filtered_phase_arr

    return ret


def make_hilbert_df(x):
    """
    Extracts analytic signal and stores components into a DataFrame
    - Real and imaginary components
    - Angle, both wrapped and unwrapped

    :param x:
    :return:
    """
    x_hil = sig.hilbert(x)

    x_hil_real = x_hil.real
    x_hil_imag = x_hil.imag

    x_hil_angle = np.angle(x_hil)
    x_hil_phase = np.unwrap(x_hil_angle)

    hilbert_df = pd.DataFrame(dict(z_t_real=x_hil_real, z_t_imag=x_hil_imag,
                                   z_t_angle=x_hil_angle, z_t_unwrap=x_hil_phase,
                                   envelope=np.abs(x_hil)
                                   ),
                              index=x.index)
    hilbert_df['signal'] = x
    return hilbert_df
