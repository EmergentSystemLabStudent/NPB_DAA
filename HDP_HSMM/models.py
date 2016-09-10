from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import itertools, collections, operator, random, copy
from basic.abstractions import ModelGibbsSampling, ModelEM, ModelMAPEM
from basic.util import engine_global_namespace
from internals import states, transitions


#########################################################
#  HDP-HMM model classes  #
#########################################################
# TODO think about factoring out base classes for HMMs and HSMMs
# TODO maybe states classes should handle log_likelihood and predictive
# likelihood methods
# TODO generate_obs should be here, not in states.py
class HMM(ModelGibbsSampling, ModelEM, ModelMAPEM):
    _states_class = states.HMMStatesPython
    _trans_class = transitions.HDPHMMTransitions
    _trans_class_conc_class = transitions.HDPHMMTransitionsConcResampling
    _init_steady_state_class = states.SteadyState
    def __init__(self,
            obs_distns,
            trans_distn=None,
            alpha=None,gamma=None,
            alpha_a_0=None,alpha_b_0=None,gamma_a_0=None,gamma_b_0=None,
            init_state_distn=None,init_state_concentration=None):
        self.state_dim = len(obs_distns)
        self.obs_distns = obs_distns
        self.states_list = []
        assert (trans_distn is not None) ^ \
                (alpha is not None and gamma is not None) ^ \
                (alpha_a_0 is not None and alpha_b_0 is not None
                        and gamma_a_0 is not None and gamma_b_0 is not None)
        if trans_distn is not None:
            self.trans_distn = trans_distn
        elif alpha is not None:
            self.trans_distn = self._trans_class(
                    state_dim=self.state_dim,
                    alpha=alpha,gamma=gamma)
        else:
            self.trans_distn = self._trans_class_conc_class(
                    state_dim=self.state_dim,
                    alpha_a_0=alpha_a_0,alpha_b_0=alpha_b_0,
                    gamma_a_0=gamma_a_0,gamma_b_0=gamma_b_0)
        if init_state_distn is not None:
            self.init_state_distn = init_state_distn
        elif init_state_concentration is not None:
            self.init_state_distn = states.InitialState(
                    state_dim=self.state_dim,
                    rho=init_state_concentration)
        else:
            # in this case, the initial state distribution is just the
            # steady-state of the transition matrix
            self.init_state_distn = self._init_steady_state_class(self)
    @property
    def stateseqs(self):
        'a convenient reference to the state sequence arrays'
        return [s.stateseq for s in self.states_list]
    @property
    def Viterbi_stateseqs(self):
        current_stateseqs = [s.stateseq for s in self.states_list]
        for s in self.states_list:
            s.Viterbi()
        ret = [s.stateseq for s in self.states_list]
        for s,seq in zip(self.states_list,current_stateseqs):
            s.stateseq = seq
        return ret
    def add_data(self,data,stateseq=None,**kwargs):
        self.states_list.append(self._states_class(model=self,data=data,
            stateseq=stateseq,**kwargs))
    def heldout_viterbi(self,data,**kwargs):
        self.add_data(data=data,stateseq=np.zeros(len(data)),**kwargs)
        s = self.states_list.pop()
        s.Viterbi()
        return s.stateseq
    def heldout_state_marginals(self,data,**kwargs):
        self.add_data(data=data,stateseq=np.zeros(len(data)),**kwargs)
        s = self.states_list.pop()
        log_margs = s.messages_forwards() + s.messages_backwards()
        log_margs -= log_margs.max(1)[:,None]
        margs = np.exp(log_margs)
        margs /= margs.sum(1)[:,None]
        return margs
    def log_likelihood(self,data=None,**kwargs):
        if data is not None:
            self.add_data(data=data,stateseq=np.zeros(len(data)),**kwargs)
            s = self.states_list.pop()
            betal = s.messages_backwards()
            return np.logaddexp.reduce(np.log(self.init_state_distn.pi_0) + betal[0] + s.aBl[0])
        else:
            if hasattr(self,'_last_resample_used_temp') and self._last_resample_used_temp:
                self._clear_caches()
            initials = np.vstack([
                s.messages_backwards()[0] + s.aBl[0] + np.log(s.pi_0)
                for s in self.states_list])
            return np.logaddexp.reduce(initials,axis=1).sum()
    def predictive_likelihoods(self,test_data,forecast_horizons,**kwargs):
        s = self._states_class(model=self,data=np.asarray(test_data),
                stateseq=np.zeros(test_data.shape[0]), # placeholder
                **kwargs)
        alphal = s.messages_forwards()
        cmaxes = alphal.max(axis=1)
        scaled_alphal = np.exp(alphal - cmaxes[:,None])
        prev_k = 0
        outs = []
        for k in forecast_horizons:
            step = k - prev_k
            cmaxes = cmaxes[:-step]
            scaled_alphal = scaled_alphal[:-step].dot(np.linalg.matrix_power(s.trans_matrix,step))
            future_likelihoods = np.logaddexp.reduce(
                    np.log(scaled_alphal) + cmaxes[:,None] + s.aBl[k:],axis=1)
            past_likelihoods = np.logaddexp.reduce(alphal[:-k],axis=1)
            outs.append(future_likelihoods - past_likelihoods)
            prev_k = k
        return outs
    def block_predictive_likelihoods(self,test_data,blocklens,**kwargs):
        s = self._states_class(model=self,data=np.asarray(test_data),
                stateseq=np.zeros(test_data.shape[0]), # placeholder
                **kwargs)
        alphal = s.messages_forwards()
        outs = []
        for k in blocklens:
            outs.append(np.logaddexp.reduce(alphal[k:],axis=1)
                    - np.logaddexp.reduce(alphal[:-k],axis=1))
        return outs
    ### generation
    def generate(self,T,keep=True,**kwargs):
        tempstates = self._states_class(model=self,T=T,initialize_from_prior=True,**kwargs)
        return self._generate(tempstates,keep)
    def _generate(self,tempstates,keep):
        obs,labels = tempstates.generate_obs(), tempstates.stateseq
        if keep:
            tempstates.added_with_generate = True
            tempstates.data = obs
            self.states_list.append(tempstates)
        return obs, labels
    ### caching
    def _clear_caches(self):
        for s in self.states_list:
            s.clear_caches()
        if hasattr(self.init_state_distn,'clear_caches'):
            self.init_state_distn.clear_caches()
    def __getstate__(self):
        self._clear_caches()
        return self.__dict__.copy()
    ### Gibbs sampling
    def resample_model(self,temp=None):
        self._last_resample_used_temp = temp is not None and temp != 1
        self.resample_obs_distns()
        self.resample_trans_distn()
        self.resample_init_state_distn()
        self.resample_states(temp=temp)
    def resample_obs_distns(self):
        # TODO TODO get rid of logical indexing! it copies data!
        for state, distn in enumerate(self.obs_distns):
            distn.resample([s.data[s.stateseq == state] for s in self.states_list])
        self._clear_caches()
    def resample_trans_distn(self):
        self.trans_distn.resample([s.stateseq for s in self.states_list])
        self._clear_caches()
    def resample_init_state_distn(self):
        self.init_state_distn.resample([s.stateseq[:1] for s in self.states_list])
        self._clear_caches()
    def resample_states(self,temp=None):
        for s in self.states_list:
            s.resample(temp=temp)
    def copy_sample(self):
        new = copy.copy(self)
        new.obs_distns = [o.copy_sample() for o in self.obs_distns]
        new.trans_distn = self.trans_distn.copy_sample()
        new.init_state_distn = self.init_state_distn.copy_sample()
        new.states_list = [s.copy_sample(new) for s in self.states_list]
        return new
    ### parallel
    def add_data_parallel(self,data,broadcast=False,**kwargs):
        import parallel
        self.add_data(data=data,**kwargs)
        if broadcast:
            parallel.broadcast_data(self.states_list[-1].data)
        else:
            parallel.add_data(self.states_list[-1].data)
    def resample_model_parallel(self,numtoresample='all',temp=None):
        if numtoresample == 'all':
            numtoresample = len(self.states_list)
        elif numtoresample == 'engines':
            import parallel
            numtoresample = min(parallel.get_num_engines(),len(self.states_list))
        ### resample parameters locally
        self.resample_obs_distns_parallel() # doesn't necessarily run parallel
        self.resample_trans_distn()
        self.resample_init_state_distn()
        ### resample states in parallel
        # choose which sequences to resample
        if numtoresample != 'all':
            added_order = {s:i for i,s in enumerate(self.states_list)}
            states_to_resample = random.sample(self.states_list,numtoresample)
            states_to_hold_out = [s for s in self.states_list if s not in states_to_resample]
            self.states_list = states_to_resample
        # actually resample the states
        self.resample_states_parallel(temp=temp)
        # add back the held-out states
        if numtoresample != 'all':
            self.states_list.extend(states_to_hold_out)
            self.states_list.sort(key=added_order.__getitem__)
    def resample_obs_distns_parallel(self):
        # this method is broken out so that it can be overridden
        # data probably needs to be broadcasted to resample in parallel
        self.resample_obs_distns()
    def resample_states_parallel(self,temp=None):
        import parallel
        states = self.states_list
        self.states_list = [] # removed because we push the global model
        raw = parallel.map_on_each(
                self._state_sampler,
                [s.data for s in states],
                kwargss=self._get_parallel_kwargss(states),
                engine_globals=dict(global_model=self,temp=temp),
                )
        self.states_list = states
        for s, stateseq in zip(self.states_list,raw):
            s.stateseq = stateseq
    def _get_parallel_kwargss(self,states_objs):
        # this method is broken out so that it can be overridden
        return [{}]*len(states_objs)
    @staticmethod
    @engine_global_namespace # access to engine globals
    def _state_sampler(data,**kwargs):
        # expects globals: global_model, temp
        global_model.add_data(data=data,initialize_from_prior=False,temp=temp,**kwargs)
        return global_model.states_list.pop().stateseq
    ### EM
    def EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run EM'
        self._clear_caches()
        ## E step
        for s in self.states_list:
            s.E_step()
        ## M step
        # observation distribution parameters
        for state, distn in enumerate(self.obs_distns):
            distn.max_likelihood([s.data for s in self.states_list],
                    [s.expectations[:,state] for s in self.states_list])
        # initial distribution parameters
        self.init_state_distn.max_likelihood(
                None, # placeholder, "should" be np.arange(self.state_dim)
                [s.expectations[0] for s in self.states_list])
        # transition parameters (requiring more than just the marginal expectations)
        self.trans_distn.max_likelihood(None,[(s.alphal,s.betal,s.aBl) for s in self.states_list])
    def Viterbi_EM_fit(self, tol=0.1, maxiter=20):
        return self.MAP_EM_fit(tol, maxiter)
    def MAP_EM_step(self):
        return self.Viterbi_EM_step()
    def Viterbi_EM_step(self):
        assert len(self.states_list) > 0, 'Must have data to run Viterbi EM'
        self._clear_caches()
        ## Viterbi step
        for s in self.states_list:
            s.Viterbi()
        ## M step
        # observation distribution parameters
        for state, distn in enumerate(self.obs_distns):
            # TODO TODO get rid of logical indexing
            distn.max_likelihood([s.data[s.stateseq == state] for s in self.states_list])
        # initial distribution parameters
        self.init_state_distn.max_likelihood(
                np.array([s.stateseq[0] for s in self.states_list]))
        # transition parameters (requiring more than just the marginal expectations)
        self.trans_distn.max_likelihood([s.stateseq for s in self.states_list])
    @property
    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) + self.state_dim**2
    def BIC(self,data=None):
        '''
        BIC on the passed data. If passed data is None (default), calculates BIC
        on the model's assigned data
        '''
        # NOTE: in principle this method computes the BIC only after finding the
        # maximum likelihood parameters (or, of course, an EM fixed-point as an
        # approximation!)
        assert data is None and len(self.states_list) > 0, 'Must have data to get BIC'
        if data is None:
            return -2*sum(self.log_likelihood(s.data).sum() for s in self.states_list) + \
                        self.num_parameters() * np.log(sum(s.data.shape[0] for s in self.states_list))
        else:
            return -2*self.log_likelihood(data) + self.num_parameters() * np.log(data.shape[0])
    ### plotting
    def _get_used_states(self,states_objs=None):
        if states_objs is None:
            states_objs = self.states_list
        canonical_ids = collections.defaultdict(itertools.count().next)
        for s in states_objs:
            for state in s.stateseq:
                canonical_ids[state]
        return map(operator.itemgetter(0),sorted(canonical_ids.items(),key=operator.itemgetter(1)))
    def _get_colors(self,states_objs=None):
        states = self._get_used_states(states_objs)
        numstates = len(states)
        return dict(zip(states,np.linspace(0,1,numstates,endpoint=True)))
    def plot_observations(self,colors=None,states_objs=None):
        if states_objs is None:
            states_objs = self.states_list
        if colors is None:
            colors = self._get_colors(states_objs)
        cmap = cm.get_cmap()
        used_states = self._get_used_states(states_objs)
        for state,o in enumerate(self.obs_distns):
            if state in used_states:
                o.plot(
                        color=cmap(colors[state]),
                        data=[s.data[s.stateseq == state] if s.data is not None else None
                            for s in states_objs],
                        indices=[np.where(s.stateseq == state)[0] for s in states_objs],
                        label='%d' % state)
        plt.title('Observation Distributions')
    def plot(self,color=None,legend=False):
        plt.gcf() #.set_size_inches((10,10))
        colors = self._get_colors()
        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):
            plt.subplot(2,num_subfig_cols,1+subfig_idx)
            self.plot_observations(colors=colors,states_objs=[s])
            plt.subplot(2,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=colors)
        if legend:
            plt.legend()

class HMMEigen(HMM):
    _states_class = states.HMMStatesEigen


#########################################################
#  HDP-HSMM model classes  #
#########################################################
class HSMM(HMM, ModelGibbsSampling, ModelEM, ModelMAPEM):
    _states_class = states.HSMMStatesPython
    _trans_class = transitions.HDPHSMMTransitions
    _trans_class_conc_class = transitions.HDPHSMMTransitionsConcResampling
    _init_steady_state_class = states.HSMMSteadyState
    def __init__(self,dur_distns,**kwargs):
        self.dur_distns = dur_distns
        super(HSMM,self).__init__(**kwargs)
        if isinstance(self.init_state_distn,self._init_steady_state_class):
            self.left_censoring_init_state_distn = self.init_state_distn
        else:
            self.left_censoring_init_state_distn = self._init_steady_state_class(self)
    @property
    def stateseqs_norep(self):
        return [s.stateseq_norep for s in self.states_list]
    @property
    def durations(self):
        return [s.durations for s in self.states_list]
    def add_data(self,data,stateseq=None,trunc=None,right_censoring=True,left_censoring=False,
            **kwargs):
        self.states_list.append(self._states_class(
            model=self,
            data=np.asarray(data),
            stateseq=stateseq,
            right_censoring=right_censoring,
            left_censoring=left_censoring,
            trunc=trunc,
            **kwargs))
    def log_likelihood(self,data=None,trunc=None,**kwargs):
        # NOTE: this only works with iid emissions
        if data is not None:
            self.add_data(data=data,trunc=trunc,stateseq=np.zeros(len(data)),**kwargs)
            s = self.states_list.pop()
            _, betastarl = s.messages_backwards()
            return np.logaddexp.reduce(np.log(s.pi_0) + betastarl[0])
        else:
            if hasattr(self,'_last_resample_used_temp') and self._last_resample_used_temp:
                self._clear_caches()
            initials = np.vstack([
                s.messages_backwards()[1][0] + np.log(s.pi_0) for s in self.states_list])
            return np.logaddexp.reduce(initials,axis=1).sum()
    ### generation
    def generate(self,T,keep=True,**kwargs):
        tempstates = self._states_class(model=self,T=T,initialize_from_prior=True,**kwargs)
        return self._generate(tempstates,keep)
    ### Gibbs sampling
    def resample_model(self,**kwargs):
        self.resample_dur_distns()
        super(HSMM,self).resample_model(**kwargs)
    def resample_dur_distns(self):
        # TODO TODO get rid of logical indexing
        for state, distn in enumerate(self.dur_distns):
            distn.resample_with_truncations(
                    data=
                    [s.durations_censored[s.untrunc_slice][s.stateseq_norep[s.untrunc_slice] == state]
                        for s in self.states_list],
                    truncated_data=
                    [s.durations_censored[s.trunc_slice][s.stateseq_norep[s.trunc_slice] == state]
                        for s in self.states_list])
        self._clear_caches()
    def copy_sample(self):
        new = super(HSMM,self).copy_sample()
        new.dur_distns = [d.copy_sample() for d in self.dur_distns]
        return new
    ### parallel
    def resample_model_parallel(self,*args,**kwargs):
        self.resample_dur_distns()
        super(HSMM,self).resample_model_parallel(*args,**kwargs)
    def _get_parallel_kwargss(self,states_objs):
        return [dict(trunc=s.trunc,left_censoring=s.left_censoring,
                    right_censoring=s.right_censoring) for s in states_objs]
    ### EM
    def EM_step(self):
        super(HSMM,self).EM_step()
        # M step for duration distributions
        for state, distn in enumerate(self.dur_distns):
            distn.max_likelihood(
                    None, # placeholder, "should" be [np.arange(s.T) for s in self.states_list]
                    [s.expectations[:,state] for s in self.states_list])
    def Viterbi_EM_step(self):
        super(HSMM,self).Viterbi_EM_step()
        # M step for duration distributions
        for state, distn in enumerate(self.dur_distns):
            # TODO TODO get rid of logical indexing
            distn.max_likelihood(
                    [s.durations[s.stateseq_norep == state] for s in self.states_list])
    @property
    def num_parameters(self):
        return sum(o.num_parameters() for o in self.obs_distns) \
                + sum(d.num_parameters() for d in self.dur_distns) \
                + self.state_dim**2 - self.state_dim
    ### plotting
    def plot_durations(self,colors=None,states_objs=None):
        if colors is None:
            colors = self._get_colors()
        if states_objs is None:
            states_objs = self.states_list
        cmap = cm.get_cmap()
        used_states = self._get_used_states(states_objs)
        for state,d in enumerate(self.dur_distns):
            if state in used_states:
                d.plot(color=cmap(colors[state]),
                        data=[s.durations[s.stateseq_norep == state]
                            for s in states_objs])
        plt.title('Durations')
    def plot(self,color=None):
        plt.gcf() #.set_size_inches((10,10))
        colors = self._get_colors()
        num_subfig_cols = len(self.states_list)
        for subfig_idx,s in enumerate(self.states_list):
            plt.subplot(3,num_subfig_cols,1+subfig_idx)
            self.plot_observations(colors=colors,states_objs=[s])
            plt.subplot(3,num_subfig_cols,1+num_subfig_cols+subfig_idx)
            s.plot(colors_dict=colors)
            plt.subplot(3,num_subfig_cols,1+2*num_subfig_cols+subfig_idx)
            self.plot_durations(colors=colors,states_objs=[s])
    def plot_summary(self,color=None):
        # if there are too many state sequences in states_list, make an
        # alternative plot that isn't so big
        raise NotImplementedError # TODO

class HSMMEigen(HSMM):
    _states_class = states.HSMMStatesEigen
