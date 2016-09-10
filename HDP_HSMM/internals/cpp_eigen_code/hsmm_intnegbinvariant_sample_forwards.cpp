using namespace Eigen;
using namespace std;

// inputs
Map<ArrayXXd> eAT(A,M,M);
Map<ArrayXXd> eaBl(aBl,M,T);
Map<ArrayXXd> ebetal(betal,rtot,T);
Map<ArrayXXd> esuperbetal(superbetal,M,T);

// locals
int t, state, substate, end;
double total, pi;
ArrayXd logdomain(M);
ArrayXd nextstate_unsmoothed(M);
ArrayXd nextstate_distr(M);
ArrayXd pair(2);

// code!

// sample first state
// logdomain = esuperbetal.col(0) + eaBl.col(0);
// nextstate_distr = (logdomain - logdomain.maxCoeff()).exp() * epi0;
// total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
// for (state=0; (total -= nextstate_distr(state)) > 0; state++) ;
// stateseq[0] = state;

t = 0;

state = initial_superstate;
substate = initial_substate;
pi = ps[state];
end = end_indices[state];

while (t < T) {
    // loop inside the substates

    while ((substate < end) && (t < T)) {
        pair = eaBl(state,t) + ebetal.col(t).segment(substate,2);
        pair = (pair - pair.maxCoeff()).exp();
        total = (1.0-pi)*pair(1) / ((1.0-pi)*pair(1) + pi*pair(0));
        substate += (((double)random())/((double)RAND_MAX)) < total;

        stateseq[t] = state;
        t += 1;
    }

    // sample the 'end' row just like a regular HMM transition

    nextstate_unsmoothed = eAT.col(state);
    int current_state = state;
    while ((state == current_state) && (t < T)) {
        logdomain = esuperbetal.col(t) + eaBl.col(t);
        logdomain(state) = ebetal(end,t) + eaBl(state,t);
        nextstate_distr = (logdomain - logdomain.maxCoeff()).exp() * nextstate_unsmoothed;

        total = nextstate_distr.sum() * (((double)random())/((double)RAND_MAX));
        for (state=0; (total -= nextstate_distr(state)) > 0; state++) ;

        stateseq[t] = current_state;
        t += 1;
    }

    substate = start_indices[state];
    end = end_indices[state];
    pi = ps[state];
}

