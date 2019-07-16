#ifndef HLM_H
#define HLM_H

#include <Eigen/Core>
#include <iostream> // cout, endl
#include <algorithm> // min
#include <math.h>

#include "util.h"
#include "nptypes.h"

namespace hlm
{
    using namespace std;
    using namespace util;
    using namespace Eigen;
    using namespace nptypes;

    template <typename Type>
    void messages_backwards_log(
      int T, int N, int P, int Lmax, int Ls[], int cLs[],
      Type *Al, Type *aDl,
      Type *aBl, Type* alDl,
      int words[], int itrunc,
      Type *betal, Type *betastarl)
    {
      int tsize;
      Type cmax;
      Type ctmp;
      NPArray<Type> eAl(Al, N, N);
      NPArray<Type> eaDl(aDl, T, N);
      NPArray<Type> eaBl(aBl, T, P);
      NPArray<Type> ealDl(alDl, T, P);

      NPArray<Type> ebetal(betal, T, N);
      NPArray<Type> ebetastarl(betastarl, T, N);

      //NPArray<Type> ealphal(itrunc, Lmax);
#ifdef HLM_TEMPS_ON_HEAP
      Array<Type, 1, Dynamic> sumsofar_alpha(itrunc);
      Array<Type, 1, Dynamic> result_alpha(itrunc);
      Array<Type, Dynamic, Dynamic> ealphal(itrunc, Lmax);
      Array<Type, Dynamic, Dynamic> cum_ealphal(itrunc, N);
      Array<Type, 1, Dynamic> result(N);
      Array<Type, 1, Dynamic> maxes(N);
#else
      Type sumsofar_alpha_buf[itrunc] __attribute__((aligned(16)));
      NPRowVectorArray<Type> sumsofar_alpha(sumsofar_alpha_buf, itrunc);
      Type result_alpha_buf[itrunc] __attribute__((aligned(16)));
      NPRowVectorArray<Type> result_alpha(result_alpha_buf, itrunc);
      Type ealphal_buf[itrunc*Lmax] __attribute__((aligned(16)));
      NPArray<Type> ealphal(ealphal_buf, itrunc, Lmax);
      Type cum_ealphal_buf[itrunc*N] __attribute__((aligned(16)));
      NPArray<Type> cum_ealphal(cum_ealphal_buf, itrunc, N);
      Type result_buf[N] __attribute__((aligned(16)));
      NPRowVectorArray<Type> result(result_buf, N);
      Type maxes_buf[N] __attribute__((aligned(16)));
      NPRowVectorArray<Type> maxes(maxes_buf, N);
#endif

      //initialize.
      Type neg_inf = -1.0*numeric_limits<Type>::infinity();
      ebetal.setConstant(neg_inf);
      ebetastarl.setConstant(neg_inf);
      ebetal.row(T-1).setZero();

      for(int t=T-1; t>=0; t--){
        tsize = min(itrunc, T-t);
        for(int i=0; i<N; i++){
          ealphal.setConstant(neg_inf);
          ctmp = 0.0;
          for(int tt=0; tt<tsize-Ls[i]+1; tt++){
            ctmp = ctmp + eaBl(t+tt, words[cLs[i]]);
            ealphal(tt, 0) = ctmp + ealDl(tt, words[cLs[i]]);
          }
          for(int j=0; j<Ls[i]-1; j++){
            sumsofar_alpha.setZero();
            for(int tt=0; tt<tsize-Ls[i]+1; tt++){
              for(int tau=0; tau<=tt; tau++){
                sumsofar_alpha(tau) = sumsofar_alpha(tau) + eaBl(t+tt+j+1, words[cLs[i]+j+1]);
                result_alpha(tau) = sumsofar_alpha(tau) + ealDl(tt-tau, words[cLs[i]+j+1]) + ealphal(j+tau, j);
              }
              cmax = result_alpha.head(tt+1).maxCoeff();
              ealphal(tt+j+1, j+1) = log((result_alpha.head(tt+1) - cmax).exp().sum()) + cmax;
              if(ealphal(tt+j+1, j+1) != ealphal(tt+j+1, j+1)){
                ealphal(tt+j+1, j+1) = neg_inf;
              }
            }
          }
          cum_ealphal.col(i) = ealphal.col(Ls[i]-1);
        }

        for(int tau=0; tau<tsize; tau++){
          result = ebetal.row(t+tau) + cum_ealphal.row(tau) + eaDl.row(tau);
          maxes = ebetastarl.row(t).cwiseMax(result);
          ebetastarl.row(t) = ((ebetastarl.row(t) - maxes).exp() + (result - maxes).exp()).log() + maxes;
          for(int nu=0; nu<N; nu++){
            if(ebetastarl(t, nu) != ebetastarl(t, nu)){
              ebetastarl(t, nu) = neg_inf;
            }
          }

        }
        if(likely(t > 0)){
          for(int nu=0; nu<N; nu++){
            result = ebetastarl.row(t) + eAl.row(nu);
            cmax = result.maxCoeff();
            ebetal(t-1, nu) = log((result - cmax).exp().sum()) + cmax;
            if(ebetal(t-1, nu) != ebetal(t-1, nu)){
              ebetal(t-1, nu) = neg_inf;
            }
          }
        }
      }
    }

}

// NOTE: this class exists for cyhton binding convenience

template <typename FloatType, typename IntType = int32_t>
class hlmc
{
    public:

    static void messages_backwards_log(
      int T, int N, int P, int Lmax, int Ls[], int cLs[],
      FloatType *Al, FloatType *aDl,
      FloatType *aBl, FloatType* alDl,
      int words[], int itrunc,
      FloatType *betal, FloatType *betastarl)
    { hlm::messages_backwards_log(T, N, P, Lmax, Ls, cLs, Al, aDl, aBl, alDl, words, itrunc, betal, betastarl); }
};

#endif
