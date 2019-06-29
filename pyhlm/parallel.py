# NOTE: pass arguments through global variables instead of arguments to exploit
# the fact that they're read-only and multiprocessing/joblib uses fork

model = None
args = None

def _get_sampled_stateseq_norep_and_durations_censored(idx):
    grp = args[idx]

    if len(grp) == 0:
        return []

    datas, kwargss = zip(*grp)

    states_list = []
    for data, kwargs in zip(datas,kwargss):
        model.add_data(data, initialize_from_prior=False, **kwargs)
        states_list.append(model.states_list.pop())

    return [(s.stateseq, s.stateseq_norep, s.durations_censored, s.log_likelihood()) for s in states_list]
