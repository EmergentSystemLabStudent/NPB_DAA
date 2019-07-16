#ifndef INTERNAL_HSMM_H
#define INTERNAL_HSMM_H

#include <Eigen/Core>
#include <iostream> // cout, endl
#include <algorithm> // min
#include <math.h>

#include "util.h"
#include "nptypes.h"

namespace internal_hsmm
{
    using namespace std;
    using namespace Eigen;
    using namespace nptypes;

    template <typename Type>
    void internal_hsmm_messages_forwards_log(
      int T, int L, int P,
      Type *aBl, Type* alDl, int word[],
      Type *alphal)
    {
      // T: Length of observations.
      // P: Number of phonemes in model. (Number of upper limit of phonemes.)
      // L: Length of the word. (Number of letters in word.)
      NPArray<Type> eaBl(aBl, T, P);
      NPArray<Type> ealDl(alDl, T, P);

      NPArray<Type> ealphal(alphal, T, L);

#ifdef HLM_TEMPS_ON_HEAP
      Array<Type,1,Dynamic> sumsofar(T-L+1);
      Array<Type,1,Dynamic> result(T-L+1);
#else
      Type sumsofar_buf[T-L+1] __attribute__((aligned(16)));
      NPRowVectorArray<Type> sumsofar(sumsofar_buf,T-L+1);
      Type result_buf[T-L+1] __attribute__((aligned(16)));
      NPRowVectorArray<Type> result(result_buf,T-L+1);
#endif

      //initialize.
      Type neg_inf = -1.0*numeric_limits<Type>::infinity();
      ealphal.setConstant(neg_inf);

      Type ctmp = 0.0;
      for(int t=0; t<T-L+1; t++){
        ctmp += eaBl(t, word[0]);
        ealphal(t, 0) = ctmp + ealDl(t, word[0]);
      }

      Type cmax;
      for(int j=0; j<L-1; j++){
        sumsofar.setZero();
        for(int t=0; t<T-L+1; t++){
          for(int tau=0; tau<=t; tau++){
            sumsofar(tau) = sumsofar(tau) + eaBl(t+j+1, word[j+1]);
            result(tau) = sumsofar(tau) + ealDl(t-tau, word[j+1]) + ealphal(j+tau, j);
          }
          cmax = result.head(t+1).maxCoeff();
          ealphal(t+j+1, j+1) = log((result.head(t+1) - cmax).exp().sum()) + cmax;
          if(ealphal(t+j+1, j+1) != ealphal(t+j+1, j+1)){
            ealphal(t+j+1, j+1) = neg_inf;
          }
        }
      }
    }

}

// NOTE: this class exists for cyhton binding convenience

template <typename FloatType, typename IntType = int32_t>
class internal_hsmmc
{
    public:

    static void internal_hsmm_messages_forwards_log(
      int T, int L, int P,
      FloatType *aBl, FloatType *alDl, int word[],
      FloatType *alphal)
    { internal_hsmm::internal_hsmm_messages_forwards_log(T, L, P, aBl, alDl, word, alphal); }

};

#endif
