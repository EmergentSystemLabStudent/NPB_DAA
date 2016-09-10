import numpy as np
import pickle, itertools, json, copy, os


### save dir and param
def get_result_dir(prefix):
    base, ext = os.path.splitext(os.path.basename(prefix))
    def filename_iterator():
        result_dir_template = base + "_result" + "_%d"
        for i in itertools.count(1):
            yield result_dir_template % i
    for name in filename_iterator():
        if not os.path.exists(name):
            return name
def save_parameters_scalar(path, keys, values):
    params = dict([(k, values[k]) for k in keys])
    with open(path, "w") as f:
        json.dump(params, f)
def save_parameters_multivariate(path, keys, values):
    params = dict([(k, values[k]) for k in keys])
    s = params['obs_hypparams']['sigma_0']
    m = params['obs_hypparams']['mu_0']
    params['obs_hypparams']['sigma_0']=str(params['obs_hypparams']['sigma_0'])
    params['obs_hypparams']['mu_0']=str(params['obs_hypparams']['mu_0'])
    with open(path, "w") as f:
        json.dump(params, f)
    params['obs_hypparams']['sigma_0']=s
    params['obs_hypparams']['mu_0']=m
def save_fig_title(path, keys, values):
    params = dict([(k, values[k]) for k in keys])
    with open(path, "w") as f:
        json.dump(params['filename'],f)


### save experiment result
class Result(object):
    def __init__(self, dirname, data_n):
        self._dir = dirname
        self.states = [[] for _ in range(data_n)]
        self.state_ranges = [[] for _ in range(data_n)]
        self.letters = [[] for _ in range(data_n)]
        self.word_list = []
    def dir_path(self, name):
        return os.path.join(self._dir, name)
    def save(self, model):
        self.save_sampled_states(model)
        self.save_word_list(model)
        self.save_state_ranges(model)
        self.write()
    ## loglikelihood_nakashima
    def get_loglikelihood(self,model):
        ll=[]
        for k in model.states_list:
            b=k.betal[0]
            c=np.logaddexp.reduce(b)
            ll.append(c)
        reloglid = np.logaddexp.reduce(ll)
        print reloglid
        return reloglid
    def save_loglikelihood(self,loglikelihood):
        np.savetxt(self.dir_path('loglikelihood.txt'), loglikelihood, '%.2f')
    ##-------------------------------------
    def save_sampled_states(self, model):
        for i, sampled in enumerate(model.states_list):
            self.states[i].append(sampled.stateseq)
            self.letters[i].append(sampled.letterseq)
    def save_state_ranges(self, model):
        for i, sample in enumerate(model.states_list):
            print sample.state_ranges
            self.state_ranges[i].append(copy.copy(sample.state_ranges))
    def save_word_list(self, model):
        self.word_list.append(copy.copy(model.word_list))
    def write(self):
        for idx, (states, letters) in enumerate(zip(self.states, self.letters)):
            np.savetxt(self.dir_path('sample_states_%d.txt' % idx), self.states[idx])
            np.savetxt(self.dir_path('sample_letters_%d.txt' % idx), self.letters[idx])
        with open(self.dir_path('sample_word_list.txt'), 'w') as f:
            pickle.dump(self.word_list, f)
        for idx, ranges in enumerate(self.state_ranges):
            with open(self.dir_path('state_ranges_%d.txt' % idx), 'w') as f:
                pickle.dump(ranges, f)
