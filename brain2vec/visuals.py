import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def pca_and_scatter_plot(_df, c=None, **kws):
    from sklearn.decomposition import PCA

    _pca_arr = PCA(2).fit_transform(_df)

    _pca_df = pd.DataFrame(_pca_arr,
                           columns=[f"PCA-{n}" for n in range(2)])


    ax = _pca_df.plot.scatter(x=0, y=1, c=c, sharex=False, **kws)
    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    return _pca_df, fig, ax


def pca_and_pair_plot(_df, _y_s=None, n_components=3, **pair_plt_kws):
    from sklearn.decomposition import PCA
    import seaborn as sns

    _pca_arr = PCA(n_components).fit_transform(_df)

    _pca_df = pd.DataFrame(_pca_arr,
                           columns=[f"PCA-{n}" for n in range(n_components)])

    plt_kws = dict(  # hue='target_val',
        diag_kws=dict(common_norm=False), kind='hist',
        diag_kind='kde', )
    _plt_df = _pca_df
    if _y_s is not None:
        _plt_df = _plt_df.join(_y_s)
        plt_kws['hue'] = _y_s.name

    plt_kws.update(pair_plt_kws)
    g = sns.pairplot(_plt_df, **plt_kws)

    # ax = _pca_df.plot.scatter(x=0, y=1, alpha=0.3, c=all_y_s, cmap='tab10', sharex=False)
    fig = g.fig  # ax.get_figure()
    fig.patch.set_facecolor('white')
    return _pca_df, fig, g

def scatter3d(*dims, alpha=0.3, c=None):
    fig = matplotlib.pyplot.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection='3d')
    assert len(dims) == 3, f"Expected 3 arrays passed in first parameters - got {len(dims)}"
    #plt_df = _pca_df.sample(10000)
    #ax.scatter(plt_df[0], plt_df[1], plt_df[2], alpha=0.3, c = all_y_s.loc[plt_df.index])
    ax.scatter(*dims, alpha=alpha, c=c)
    return fig, ax


def plot_word_sample_region(data_map, word_code=None, figsize=(15, 5), plot_features=False,
                            subplot_kwargs=None, feature_key='ecog', feature_ax=None, ax=None):
    word_code = np.random.choice(list(data_map['word_code_d'].keys())) if word_code is None else word_code

    t_silence_ixes = data_map['sample_index_map'][-word_code]
    t_speaking_ixes = data_map['sample_index_map'][word_code]

    silence_min_ix, silence_max_ix = t_silence_ixes[0].min(), t_silence_ixes[-1].max()
    speaking_min_ix, speaking_max_ix = t_speaking_ixes[0].min(), t_speaking_ixes[-1].max()

    padding = pd.Timedelta(.75, 's')

    plt_min = min(silence_min_ix, speaking_min_ix) - padding
    plt_max = max(silence_max_ix, speaking_max_ix) + padding
    plt_len = plt_max - plt_min

    #####
    plt_audio = (data_map['audio'].loc[plt_min:plt_max]
                 # .resample('5ms').first().fillna(method='ffill'),
                 .resample('5ms').median().fillna(0)
                 # .resample('5ms').interpolate().fillna(0)
                 )

    silence_s = pd.Series(0, index=plt_audio.index)
    silence_s.loc[silence_min_ix: silence_max_ix] = 0.95

    speaking_s = pd.Series(0, index=plt_audio.index)
    speaking_s.loc[speaking_min_ix: speaking_max_ix] = 0.95

    #####
    # feature_ax = None
    splt_kws = dict() if subplot_kwargs is None else subplot_kwargs
    if not plot_features and ax is None:
        fig, ax = matplotlib.pyplot.subplots(figsize=figsize, **splt_kws)
    elif not plot_features:
        fig = ax.get_figure()
    elif plot_features and ax is None or feature_ax is None:
        fig, (ax, feature_ax) = matplotlib.pyplot.subplots(figsize=figsize, nrows=2, **splt_kws)
    else:
        fig = ax.get_figure()

    ax = plt_audio.plot(legend=False, alpha=0.4, color='tab:grey', label='audio', ax=ax)
    ax.set_title(f"Min-ts={plt_min} || Max-ts={plt_max}\n\
    Labeled Regions: word_code={word_code}, word='{data_map['word_code_d'][word_code]}'\
    \nSpeaking N windows={len(t_speaking_ixes)}; Silence N windows={len(t_speaking_ixes)}")
    ax2 = ax.twinx()

    ax2.set_ylim(0.05, 1.1)
    # ax.axvline(silence_min_ix / pd.Timedelta(1,'s'))
    # (data_map['stim'].reindex(data_map['audio'].index).fillna(method='ffill').loc[plt_min: plt_max] > 0).astype(
    #    int).plot(ax=ax2, color='tab:blue', label='original stim')
    (data_map['stim'].resample('5ms').first().fillna(method='ffill').loc[plt_min: plt_max] > 0).astype(
        int).plot(ax=ax2, color='tab:blue', label='stim')

    silence_s.plot(ax=ax2, color='red', lw=4, label='silence')
    speaking_s.plot(ax=ax2, color='green', lw=4, label=f"speaking ")
    ax.legend()
    ax2.legend()

    if feature_ax is not None:
        feat_df = data_map[feature_key].loc[plt_min: plt_max]
        feat_df.plot(ax=feature_ax,
                     cmap='viridis', grid=True,
                     alpha=0.44, legend=False)
        feature_ax.set_title(f"Features\nplot shape={feat_df.shape}); window length={len(t_speaking_ixes[0])}")

    fig.tight_layout()
    return fig, ax


def plot_region_over_signal(signal_s, region_min, region_max,
                            padding_time=pd.Timedelta('1s'),
                            plot_signal=True,
                            ax=None, signal_plot_kwargs=None, region_plot_kwargs=None):
    def_signal_plot_kwargs = dict(color='tab:green', alpha=0.5)
    if isinstance(signal_plot_kwargs, dict):
        def_signal_plot_kwargs.update(signal_plot_kwargs)
    elif signal_plot_kwargs is not None:
        raise ValueError()

    signal_plot_kwargs = def_signal_plot_kwargs

    region_plot_kwargs = dict() if region_plot_kwargs is None else region_plot_kwargs

    plt_min = region_min - padding_time
    # print(f"{plt_min} = {region_min} - {padding_time}")

    plt_max = region_max + padding_time
    # print(f"{plt_max} = {region_max} + {padding_time}")

    signal_s = signal_s.loc[plt_min: plt_max]
    plt_ix = signal_s.index

    region_line_s = pd.Series(0, index=plt_ix)
    region_line_s.loc[region_min: region_max] = 1

    ax2 = ax
    if plot_signal:
        ax = signal_s.loc[plt_min:plt_max].plot(ax=ax, **signal_plot_kwargs)
        ax2 = ax.twinx()

    ax2 = region_line_s.loc[plt_min:plt_max].plot(ax=ax2, **region_plot_kwargs)

    fig = ax.get_figure()
    fig.patch.set_facecolor('white')
    return fig, ax, ax2


def plot_multi_region_over_signal(signal_s, region_min_max_tuples,  # region_min, region_max,
                            padding_time=pd.Timedelta('1s'),
                            plot_signal=True,
                            ax=None, signal_plot_kwargs=None, region_plot_kwargs=None):
    def_signal_plot_kwargs = dict(color='tab:green', alpha=0.5)
    if isinstance(signal_plot_kwargs, dict):
        def_signal_plot_kwargs.update(signal_plot_kwargs)
    elif signal_plot_kwargs is not None:
        raise ValueError()

    signal_plot_kwargs = def_signal_plot_kwargs

    region_plot_kwargs = dict() if region_plot_kwargs is None else region_plot_kwargs

    region_min = min(t_ for t_, _t, _, _ in region_min_max_tuples)
    region_max = max(_t for t_, _t, _, _ in region_min_max_tuples)

    plt_min = region_min - padding_time

    plt_max = region_max + padding_time

    signal_s = signal_s.loc[plt_min: plt_max]
    plt_ix = signal_s.index

    ax2 = ax
    if plot_signal:
        ax = signal_s.loc[plt_min:plt_max].plot(ax=ax, **signal_plot_kwargs)
        ax2 = ax.twinx()

    region_lines_l = list()
    for t_, _t, label, v in region_min_max_tuples:
        region_line_s = pd.Series(0, index=plt_ix, name=label)
        region_line_s.loc[t_: _t] = v
        region_lines_l.append(region_line_s)

    region_df = pd.concat(region_lines_l, axis=1)

    ax2 = region_df.plot(ax=ax2, **region_plot_kwargs)

    fig = ax.get_figure()
    fig.patch.set_facecolor('white')

    return fig, ax, ax2


# From https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    fig = axes[0].get_figure()
    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    axes[0].fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    axes[0].plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    axes[0].plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, "o-")
    axes[1].fill_between(
        train_sizes,
        fit_times_mean - fit_times_std,
        fit_times_mean + fit_times_std,
        alpha=0.1,
    )
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    axes[2].grid()
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
    axes[2].fill_between(
        fit_time_sorted,
        test_scores_mean_sorted - test_scores_std_sorted,
        test_scores_mean_sorted + test_scores_std_sorted,
        alpha=0.1,
    )
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return fig, axes