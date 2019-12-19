
N_THREADS_PER_RUN = 1


def provide_data(dataset, seed=28, n_train_tasks=None, n_samples=None):

    from experiments.data_sim import SinusoidNonstationaryDataset, MNISTRegressionDataset, PhysionetDataset, \
        GPFunctionsDataset, CauchyDataset, SinusoidDataset
    import numpy as np


    N_TEST_TASKS = 200
    N_VALID_TASKS = 200

    N_TEST_SAMPLES = 200


    """ Prepare Data """
    if dataset == 'sin-nonstat':
        dataset = SinusoidNonstationaryDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 20
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20
        
    elif dataset == 'sin':
        dataset = SinusoidDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 5
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20
        
    elif dataset == 'gp-funcs':
        dataset = GPFunctionsDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 5
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20
        
    elif dataset == 'cauchy':
        dataset = CauchyDataset(random_state=np.random.RandomState(seed + 1))

        if n_samples is None:
            n_train_samples = n_context_samples = 20
        else:
            n_train_samples = n_context_samples = n_samples

        if n_train_tasks is None: n_train_tasks = 20

    elif dataset == 'mnist':
        dataset = MNISTRegressionDataset(random_state=np.random.RandomState(seed + 1))
        N_TEST_SAMPLES = -1
        N_VALID_TASKS = N_TEST_TASKS = 1000
        n_context_samples = 200
        n_train_samples = 28*28

    elif 'physionet' in dataset:
        variable_id = int(dataset[-1])
        assert 0 <= variable_id <= 5
        dataset = PhysionetDataset(random_state=np.random.RandomState(seed + 1), variable_id=variable_id)
        n_context_samples = 24
        n_train_samples = 47

        n_train_tasks = 200
        N_VALID_TASKS = N_TEST_TASKS = 500


    else:
        raise NotImplementedError('Does not recognize dataset flag')

    data_train = dataset.generate_meta_train_data(n_tasks=n_train_tasks, n_samples=n_train_samples)

    data_test_valid = dataset.generate_meta_test_data(n_tasks=N_TEST_TASKS + N_VALID_TASKS, n_samples_context=n_context_samples,
                                                n_samples_test=N_TEST_SAMPLES)
    data_valid = data_test_valid[N_VALID_TASKS:]
    data_test = data_test_valid[:N_VALID_TASKS]

    return data_train, data_valid, data_test


def select_best_configs(analysis, metric, mode='max', N=5):
    rows = analysis._retrieve_rows(metric='test_ll', mode='max')
    all_configs = analysis.get_all_configs()
    reverse = mode == 'max'
    best_paths = sorted(rows, key=lambda k: rows[k][metric], reverse=reverse)[:N]
    best_configs = [all_configs[path] for path in best_paths]
    return best_configs
