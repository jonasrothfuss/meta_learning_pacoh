from experiments.data_sim import provide_data
import time
import numpy as np

N_ITER = 5

DATASETS = ['sin_20']#['sin_20', 'cauchy_20', 'swissfel', 'physionet_0', 'physionet_1']

compute_times_meta_train = dict([(ds, {}) for ds in DATASETS])
compute_times_meta_test = dict([(ds, {}) for ds in DATASETS])

METHODS = ['pacoh_map', 'pacoh_svgd', 'pacoh_vi', 'mlap']

NN_LAYERS = [32, 32, 32, 32]


for dataset in DATASETS:
    meta_train_data, _, meta_test_data = provide_data('sin_20')


    from meta_learn.GPR_meta_mll import GPRegressionMetaLearned

    model_map = GPRegressionMetaLearned(meta_train_data, num_iter_fit=1000,
                                           covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
                                           kernel_nn_layers=NN_LAYERS, task_batch_size=len(meta_train_data))

    from meta_learn.GPR_meta_svgd import GPRegressionMetaLearnedSVGD

    model_svgd = GPRegressionMetaLearnedSVGD(meta_train_data, num_iter_fit=1000, num_particles=5,
                                           covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
                                           kernel_nn_layers=NN_LAYERS,
                                           bandwidth=0.5)

    from meta_learn.GPR_meta_vi import GPRegressionMetaLearnedVI

    model_vi = GPRegressionMetaLearnedVI(meta_train_data, num_iter_fit=1000,  covar_module='NN', mean_module='NN',
                                         mean_nn_layers=NN_LAYERS, svi_batch_size=5,
                                         kernel_nn_layers=NN_LAYERS, cov_type='diag', normalize_data=True)

    from meta_learn.GPR_meta_mlap import GPRegressionMetaLearnedPAC

    model_mlap = GPRegressionMetaLearnedPAC(meta_train_data, num_iter_fit=1000,
                                          svi_batch_size=5, covar_module='NN', mean_module='NN', mean_nn_layers=NN_LAYERS,
                                          kernel_nn_layers=NN_LAYERS, cov_type='diag', normalize_data=True)

    for gp_model, model_name in [(model_map, 'pacoh_map'), (model_svgd, 'pacoh_svgd'), (model_vi, 'pacoh_vi'),
                                 (model_mlap, 'mlap')]:

        durations_meta_train, durations_meta_test = [], []
        for i in range(N_ITER):
            t = time.time()
            gp_model.meta_fit(log_period=100, n_iter=10, verbose=False)
            duration = time.time() - t
            print('duration meta-train step', duration)
            durations_meta_train.append(duration)

            t = time.time()
            if model_name == 'mlap':
                gp_model.predict(*meta_test_data[0][:3], n_iter_meta_test=1000)
            else:
                gp_model.predict(*meta_test_data[0][:3])
            duration = time.time() - t
            print('duration meta-test', duration)
            durations_meta_test.append(duration)

        compute_times_meta_train[dataset][model_name] = (np.mean(durations_meta_train) / 10., np.std(durations_meta_train) / 10.)
        compute_times_meta_test[dataset][model_name] = (np.mean(durations_meta_test) / 10., np.std(durations_meta_test) / 10.)


    import pprint
    pprint.pprint(compute_times_meta_train)

    labels = ['PACOH-MAP', "PACOH-SVGD", "PACOH-VI", "MLAP"]

    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # meta_train iter
    durations_mean, durations_std = list(zip(*[compute_times_meta_train[dataset][method] for method in METHODS]))
    ax[0].bar(range(len(labels)), durations_mean, yerr=durations_std, align='center', alpha=0.8, ecolor='black',
            capsize=10)
    ax[0].set_xticks(range(len(labels)))
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("duration per iteration (sec)")
    ax[0].set_title("Meta-train")

    durations_mean, durations_std = list(zip(*[compute_times_meta_test[dataset][method] for method in METHODS]))
    ax[1].bar(range(len(labels)), durations_mean, yerr=durations_std, align='center', alpha=0.8, ecolor='black',
              capsize=10)
    ax[1].set_xticks(range(len(labels)))
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("duration for meta-test inference (sec)")
    ax[1].set_title("Meta-test")
    ax[1].set_yscale('log')


    fig.tight_layout()
    fig.savefig('computational_comparision.pdf')
    fig.show()
