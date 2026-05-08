/*==========================================================================
 * cr3bp_stt_time_derivatives.h
 *
 * Higher-order time derivatives of the CR3BP state, STM, and 2nd-4th
 * order State Transition Tensors using truncated Taylor-series arithmetic.
 *
 * CONVENTIONS
 * -----------
 *   raw time derivative :  X^(k)   = d^k X / dt^k
 *   Taylor coefficient  :  Xc[k]   = X^(k) / k!
 *
 * Internally we work entirely with Taylor coefficients (Xc, Phi1c, ...).
 * The final output arrays are raw time derivatives  X^(k) = k! * Xc[k].
 *
 * INDEX LAYOUT  (0-based, column-major, consistent with MATLAB)
 * ---------------------------------------------------------------
 *   Phi1(i,a)           ->  i + 6*a                         (36)
 *   Phi2(i,a,b)         ->  i + 6*(a + 6*b)                 (216)
 *   Phi3(i,a,b,c)       ->  i + 6*(a + 6*(b + 6*c))         (1296)
 *   Phi4(i,a,b,c,d)     ->  i + 6*(a + 6*(b + 6*(c + 6*d))) (7776)
 *
 *   For the k-th Taylor-coefficient slice we add k*<block_size>.
 *
 * NMAX = 12  (maximum derivative order supported).
 *=========================================================================*/
#ifndef CR3BP_STT_TIME_DERIVATIVES_H
#define CR3BP_STT_TIME_DERIVATIVES_H

#define CR3BP_NMAX 12

/*=========================================================================
 * C-linkage API: callable from both C and C++ code.
 *
 * compute_cr3bp_stt_taylor_coefficients
 *
 * Given:
 *   X0[6]              state at t0
 *   Phi1_0[36]         STM at t0           (column-major 6x6)
 *   Phi2_0[216]        2nd STT at t0       (column-major 6x6x6)
 *   Phi3_0[1296]       3rd STT at t0       (column-major 6x6x6x6)
 *   Phi4_0[7776]       4th STT at t0       (column-major 6x6x6x6x6)
 *   mu                 CR3BP mass parameter
 *   N                  desired derivative order (0..NMAX)
 *
 * Outputs (caller must allocate):
 *   Xders[6*(N+1)]                  raw time derivatives X^(k)
 *   Phi1ders[36*(N+1)]              raw Phi1^(k)
 *   Phi2ders[216*(N+1)]             raw Phi2^(k)
 *   Phi3ders[1296*(N+1)]            raw Phi3^(k)
 *   Phi4ders[7776*(N+1)]            raw Phi4^(k)
 *
 * Returns 0 on success, nonzero on error (e.g. N out of range).
 *=========================================================================*/

#ifdef __cplusplus
extern "C" {
#endif

int compute_cr3bp_stt_taylor_coefficients(
    const double* X0,
    const double* Phi1_0,
    const double* Phi2_0,
    const double* Phi3_0,
    const double* Phi4_0,
    double mu,
    int N,
    double* Xders,
    double* Phi1ders,
    double* Phi2ders,
    double* Phi3ders,
    double* Phi4ders);

#ifdef __cplusplus
}
#endif

#endif /* CR3BP_STT_TIME_DERIVATIVES_H */
