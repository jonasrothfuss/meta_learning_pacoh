


def select_best_configs(analysis, metric, mode='max', N=5):
    rows = analysis._retrieve_rows(metric='test_ll', mode='max')
    all_configs = analysis.get_all_configs()
    reverse = mode == 'max'
    best_paths = sorted(rows, key=lambda k: rows[k][metric], reverse=reverse)[:N]
    best_configs = [all_configs[path] for path in best_paths]
    return best_configs
