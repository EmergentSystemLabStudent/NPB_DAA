# distutils: language = c++
# distutils: extra_compile_args = -std=c++11 -O3 -w -DNDEBUG -DHLM_TEMPS_ON_HEAP
# distutils: include_dirs = deps/
# cython: boundscheck = False
# cython: language_level=3

import numpy as np
cimport numpy as np

from libc.stdint cimport int32_t

from cython cimport floating

cdef extern from "hlm_messages.h":
    cdef cppclass hlmc[Type]:
        hlmc()
        void messages_backwards_log(
            int T, int N, int P, int Lmax, int[] Ls, int[] cLs,
            Type *Al, Type *aDl,
            Type *aBl, Type* alDl,
            int[] words, int itrunc,
            Type *betal, Type *betastarl) nogil except +

def messages_backwards_log(
        floating[:,::1] aBl not None,
        floating[:,::1] aDl not None,
        floating[:,::1] alDl not None,
        floating[:,::1] aAl not None,
        int[::1] words not None,
        int[::1] Ls not None,
        int[::1] cLs not None,
        int Lmax,
        int itrunc,
        np.ndarray[floating, ndim=2, mode="c"] betal not None,
        np.ndarray[floating, ndim=2, mode="c"] betastarl not None):

    cdef hlmc[floating] ref

    ref.messages_backwards_log(
        betal.shape[0], betal.shape[1], aBl.shape[1], Lmax, &Ls[0], &cLs[0],
        &aAl[0, 0], &aDl[0, 0],
        &aBl[0, 0], &alDl[0, 0],
        &words[0], itrunc,
        &betal[0, 0], &betastarl[0, 0]
    )

    return betal, betastarl
