/*==========================================================================
 * cr3bp_stt_time_derivatives.h
 *
 * Higher-order time derivatives of the CR3BP state, STM, and 2nd–4th
 * order State Transition Tensors using truncated Taylor-series arithmetic.
 *
 * CONVENTIONS
 * -----------
 *   raw time derivative :  X^(k)   = d^k X / dt^k
 *   Taylor coefficient  :  Xc[k]   = X^(k) / k!
 *
 * Internally we work entirely with Taylor coefficients (Xc, Phi1c, …).
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

#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>

static const int N_STATE = 6;
static const int NMAX    = 12;

/* ---- Flat-index helpers (0-based) ---- */
inline int idx2(int i, int a)
{ return i + 6*a; }

inline int idx3(int i, int a, int b)
{ return i + 6*(a + 6*b); }

inline int idx4(int i, int a, int b, int c)
{ return i + 6*(a + 6*(b + 6*c)); }

inline int idx5(int i, int a, int b, int c, int d)
{ return i + 6*(a + 6*(b + 6*(c + 6*d))); }

/*=========================================================================
 * TaylorScalar: truncated Taylor series for a single scalar quantity.
 *
 *   c[k] = f^(k)(t0) / k!     for k = 0 … Nord
 *
 * We provide: addition, subtraction, multiplication, scalar multiply,
 *             reciprocal, power (real exponent), square-root, and the
 *             composition  g = s^alpha  via a robust recurrence.
 *=========================================================================*/
struct TaylorScalar {
    int Nord;               // truncation order
    std::vector<double> c;  // Taylor coefficients c[0..Nord]

    TaylorScalar() : Nord(0), c(1, 0.0) {}

    explicit TaylorScalar(int N) : Nord(N), c(N+1, 0.0) {}

    TaylorScalar(int N, double val) : Nord(N), c(N+1, 0.0) { c[0] = val; }

    /* element access */
    double  operator[](int k) const { return c[k]; }
    double& operator[](int k)       { return c[k]; }

    /* --- arithmetic --- */

    TaylorScalar operator+(const TaylorScalar& b) const {
        TaylorScalar r(Nord);
        for (int k = 0; k <= Nord; ++k) r.c[k] = c[k] + b.c[k];
        return r;
    }
    TaylorScalar operator-(const TaylorScalar& b) const {
        TaylorScalar r(Nord);
        for (int k = 0; k <= Nord; ++k) r.c[k] = c[k] - b.c[k];
        return r;
    }

    /* Cauchy product:  (a*b)_k = sum_{j=0}^k a_j b_{k-j} */
    TaylorScalar operator*(const TaylorScalar& b) const {
        TaylorScalar r(Nord);
        for (int k = 0; k <= Nord; ++k) {
            double s = 0.0;
            for (int j = 0; j <= k; ++j) s += c[j] * b.c[k-j];
            r.c[k] = s;
        }
        return r;
    }

    /* scalar multiply */
    TaylorScalar operator*(double s) const {
        TaylorScalar r(Nord);
        for (int k = 0; k <= Nord; ++k) r.c[k] = c[k] * s;
        return r;
    }
    friend TaylorScalar operator*(double s, const TaylorScalar& a) {
        return a * s;
    }

    TaylorScalar operator-() const {
        TaylorScalar r(Nord);
        for (int k = 0; k <= Nord; ++k) r.c[k] = -c[k];
        return r;
    }

    TaylorScalar& operator+=(const TaylorScalar& b) {
        for (int k = 0; k <= Nord; ++k) c[k] += b.c[k];
        return *this;
    }
    TaylorScalar& operator-=(const TaylorScalar& b) {
        for (int k = 0; k <= Nord; ++k) c[k] -= b.c[k];
        return *this;
    }

    /*------------------------------------------------------------------
     * Reciprocal: if this = a, compute r = 1/a.
     *   r[0] = 1/a[0]
     *   r[k] = -(1/a[0]) sum_{j=1}^k a[j] r[k-j]
     *------------------------------------------------------------------*/
    TaylorScalar reciprocal() const {
        TaylorScalar r(Nord);
        double inv0 = 1.0 / c[0];
        r.c[0] = inv0;
        for (int k = 1; k <= Nord; ++k) {
            double s = 0.0;
            for (int j = 1; j <= k; ++j) s += c[j] * r.c[k-j];
            r.c[k] = -inv0 * s;
        }
        return r;
    }

    /*------------------------------------------------------------------
     * Power: g = a^alpha  for real alpha.
     * Uses the standard recurrence for Taylor coefficients:
     *   g[0] = a[0]^alpha
     *   g[k] = (1 / (k * a[0])) sum_{j=1}^{k}
     *              [ (alpha*(k-j+1) - (j-1)) * a[j] * g[k-j+1-1] ]
     *
     * Cleaner form (Knuth / Brent–Kung):
     *   g[0] = a[0]^alpha
     *   g[k] = (1/(k * a[0])) sum_{j=0}^{k-1}
     *              [ (alpha*(k-j) - j) * a[k-j] * g[j] ]
     *------------------------------------------------------------------*/
    TaylorScalar power(double alpha) const {
        TaylorScalar g(Nord);
        g.c[0] = std::pow(c[0], alpha);
        double inv0 = 1.0 / c[0];
        for (int k = 1; k <= Nord; ++k) {
            double s = 0.0;
            for (int j = 0; j < k; ++j) {
                s += (alpha * (k - j) - j) * c[k - j] * g.c[j];
            }
            g.c[k] = inv0 * s / (double)k;
        }
        return g;
    }

    /* Convenience wrappers */
    TaylorScalar sqrt_ts() const { return power(0.5); }

    TaylorScalar inv_ts()  const { return reciprocal(); }
};

/*=========================================================================
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
 *=========================================================================*/
void compute_cr3bp_stt_taylor_coefficients(
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

#endif /* CR3BP_STT_TIME_DERIVATIVES_H */
