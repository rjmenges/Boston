/*==========================================================================
 * cr3bp_stt_time_derivatives.cpp
 *
 * Numerical computation of higher-order time derivatives for the CR3BP
 * state, STM (Phi1), and 2nd–4th order State Transition Tensors
 * (Phi2, Phi3, Phi4) using truncated Taylor-series arithmetic.
 *
 * ALGORITHM OVERVIEW
 * ------------------
 * 1. Build Taylor coefficient series for the state  X(t)  about t0
 *    using the CR3BP equations of motion in Taylor-coefficient form.
 *
 * 2. Build Taylor coefficient series for the spatial-derivative tensors
 *    A(t), B(t), C(t), D(t) of the CR3BP vector field, evaluated along
 *    the Taylor-expanded nominal trajectory X(t).
 *
 * 3. Propagate Taylor coefficients of Phi1, Phi2, Phi3, Phi4 using
 *    the variational equations in Taylor-coefficient form:
 *       Phi_c[k+1] = RHS_c[k] / (k+1)
 *    where RHS_c[k] is the k-th Taylor coefficient of the right-hand
 *    side, computed via Cauchy products of A/B/C/D with Phi tensors.
 *
 * 4. Convert Taylor coefficients back to raw time derivatives:
 *       X^(k) = k! * Xc[k]
 *
 * NOTATION
 * --------
 *   Xc[k]   = X^(k)/k!       Taylor coefficient of the state
 *   Ac[k]   = A^(k)/k!       Taylor coefficient of the Jacobian
 *   Phi1c[k]= Phi1^(k)/k!   Taylor coefficient of the STM
 *   etc.
 *
 * All internal indexing is 0-based. Output arrays use column-major
 * (MATLAB) layout.
 *=========================================================================*/

#include "cr3bp_stt_time_derivatives.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <stdexcept>

/* ---- Internal constants ---- */
static const int NMAX = CR3BP_NMAX;

/* ---- Flat-index helpers (0-based, column-major) ---- */
static inline int idx2(int i, int a)
{ return i + 6*a; }

static inline int idx3(int i, int a, int b)
{ return i + 6*(a + 6*b); }

static inline int idx4(int i, int a, int b, int c)
{ return i + 6*(a + 6*(b + 6*c)); }

static inline int idx5(int i, int a, int b, int c, int d)
{ return i + 6*(a + 6*(b + 6*(c + 6*d))); }

/*=========================================================================
 * TaylorScalar: truncated Taylor series for a single scalar quantity.
 *
 *   c[k] = f^(k)(t0) / k!     for k = 0 ... Nord
 *=========================================================================*/
struct TaylorScalar {
    int Nord;
    std::vector<double> c;

    TaylorScalar() : Nord(0), c(1, 0.0) {}
    explicit TaylorScalar(int N) : Nord(N), c(N+1, 0.0) {}
    TaylorScalar(int N, double val) : Nord(N), c(N+1, 0.0) { c[0] = val; }

    double  operator[](int k) const { return c[k]; }
    double& operator[](int k)       { return c[k]; }
};

/* ---- factorial table ---- */
static double fact_table[NMAX+2];
static bool   fact_init = false;

static void init_factorials() {
    if (fact_init) return;
    fact_table[0] = 1.0;
    for (int k = 1; k <= NMAX+1; ++k)
        fact_table[k] = fact_table[k-1] * k;
    fact_init = true;
}

/*=========================================================================
 * Cauchy-product helper for arrays of Taylor-coefficient tensors.
 *
 * Given two sequences of "matrix-like" Taylor coefficient arrays
 *   Ac[0..K] each of size sA
 *   Bc[0..K] each of size sB
 * compute the k-th Cauchy product entry at flat indices (ia, ib):
 *   sum_{j=0}^{k} Ac[j][ia] * Bc[k-j][ib]
 *
 * This is used heavily for the contraction  A(t)*Phi(t).
 *=========================================================================*/

/* (Cauchy product helpers are implemented inline where needed) */

/*=========================================================================
 * Build Taylor coefficient series for the CR3BP spatial derivative
 * tensors A, B, C, D as functions of the state Taylor series.
 *
 * The CR3BP accelerations depend on x, y, z through:
 *   q1 = x + mu,   q2 = x - 1 + mu
 *   s1 = q1^2 + y^2 + z^2,   s2 = q2^2 + y^2 + z^2
 *   g1 = s1^(-3/2),           g2 = s2^(-3/2)
 *
 * We also need higher inverse-power scalars for the partial derivatives:
 *   h1 = s1^(-5/2),  h2 = s2^(-5/2)     (for B)
 *   e1 = s1^(-7/2),  e2 = s2^(-7/2)     (for C)
 *   f1 = s1^(-9/2),  f2 = s2^(-9/2)     (for D)
 *
 * The Jacobian A, Hessian B, third-derivative C, and fourth-derivative D
 * of the vector field  f(X) = [vx; vy; vz; ax; ay; az]  are computed
 * component-by-component.
 *
 * Since f_1=vx, f_2=vy, f_3=vz are linear in the state, their higher
 * spatial derivatives vanish for order >=2.
 *   A(1,4)=1, A(2,5)=1, A(3,6)=1  (0-indexed: A(0,3)=1, etc.)
 *   B, C, D have zero rows for i=0,1,2.
 *
 * For i=3,4,5 (the acceleration components), the partial derivatives
 * involve products of q1, q2, y, z with the inverse-power scalars
 * g1, g2, h1, h2, e1, e2, f1, f2.
 *=========================================================================*/

/* Size constants for tensor blocks */
static const int SZ1 = 36;    /* 6x6 */
static const int SZ2 = 216;   /* 6x6x6 */
static const int SZ3 = 1296;  /* 6x6x6x6 */
static const int SZ4 = 7776;  /* 6x6x6x6x6 */

/*=========================================================================
 * MAIN FUNCTION
 *=========================================================================*/
extern "C"
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
    double* Phi4ders)
{
    init_factorials();

    if (N < 0 || N > NMAX)
        return -1;  /* error: N out of range */

    const int K = N;          /* number of derivative orders */
    const int Kp1 = K + 1;   /* number of Taylor coefficients: 0..K */
    const double mu1 = 1.0 - mu;  /* mass of primary */

    /*==================================================================
     * STEP 1: Build Taylor coefficient series for the state X(t).
     *
     * State components: x, y, z, vx, vy, vz
     * We store Taylor coefficients xc[0..K], yc[0..K], etc.
     *
     * From the CR3BP EOMs:
     *   xc[k+1]  = vxc[k] / (k+1)
     *   yc[k+1]  = vyc[k] / (k+1)
     *   zc[k+1]  = vzc[k] / (k+1)
     *   vxc[k+1] = axc[k] / (k+1)
     *   vyc[k+1] = ayc[k] / (k+1)
     *   vzc[k+1] = azc[k] / (k+1)
     *
     * where the acceleration Taylor coefficients axc, ayc, azc are
     * computed from the Taylor-series representations of the CR3BP
     * nonlinear terms.
     *==================================================================*/

    /* We need K+1 Taylor coefficients for the state (indices 0..K).
     * To get the K-th coefficient, we need acceleration coefficients
     * up to index K-1. To build those we need state coefficients up
     * to index K-1 (for the nonlinear terms).
     *
     * Actually we build state coefficients 0..K incrementally:
     * for each order k=0..K-1 we compute the acceleration at order k,
     * then advance the state to order k+1.
     */

    /* Allocate scalar Taylor series for state components */
    TaylorScalar xc(K), yc(K), zc(K), vxc(K), vyc(K), vzc(K);

    /* Set k=0 initial conditions */
    xc[0]  = X0[0]; yc[0]  = X0[1]; zc[0]  = X0[2];
    vxc[0] = X0[3]; vyc[0] = X0[4]; vzc[0] = X0[5];

    /* Intermediate Taylor series for CR3BP nonlinear terms.
     * We'll build these incrementally alongside the state. */
    TaylorScalar q1c(K), q2c(K);     /* q1 = x+mu, q2 = x-1+mu */
    TaylorScalar yy(K), zz(K);       /* y^2, z^2 */
    TaylorScalar q1q1(K), q2q2(K);   /* q1^2, q2^2 */
    TaylorScalar s1c(K), s2c(K);     /* s1 = q1^2+y^2+z^2, s2 = ... */
    TaylorScalar g1c(K), g2c(K);     /* g1 = s1^(-3/2), g2 = s2^(-3/2) */

    /* Scalars needed for spatial derivative tensors */
    TaylorScalar h1c(K), h2c(K);     /* s^(-5/2) */
    TaylorScalar e1c(K), e2c(K);     /* s^(-7/2) */
    TaylorScalar f1c(K), f2c(K);     /* s^(-9/2) */

    /* Products of coordinates with inverse-power scalars */
    TaylorScalar q1g1(K), q2g2(K);   /* q1*g1, q2*g2 */
    TaylorScalar yg1(K), yg2(K);     /* y*g1, y*g2 */
    TaylorScalar zg1(K), zg2(K);     /* z*g1, z*g2 */

    /* Build the state series incrementally.
     * At each step k, we have xc[0..k], etc. and we compute
     * the acceleration at order k, then advance to k+1. */

    /* Helper: given state coefficients up to order k, compute
     * all intermediate scalar series up to order k, then the
     * acceleration coefficients at order k. */

    auto build_intermediates_at_k = [&](int k) {
        /* q1 = x + mu  (constant shift) */
        q1c[k] = xc[k] + (k == 0 ? mu : 0.0);
        /* q2 = x - 1 + mu  (constant shift) */
        q2c[k] = xc[k] + (k == 0 ? (-1.0 + mu) : 0.0);

        /* y^2 via Cauchy product of y with y */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += yc[j] * yc[k-j];
          yy[k] = s; }

        /* z^2 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += zc[j] * zc[k-j];
          zz[k] = s; }

        /* q1^2 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += q1c[j] * q1c[k-j];
          q1q1[k] = s; }

        /* q2^2 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += q2c[j] * q2c[k-j];
          q2q2[k] = s; }

        /* s1 = q1^2 + y^2 + z^2 */
        s1c[k] = q1q1[k] + yy[k] + zz[k];

        /* s2 = q2^2 + y^2 + z^2 */
        s2c[k] = q2q2[k] + yy[k] + zz[k];
    };

    auto build_powers_at_k = [&](int k) {
        /* g1 = s1^(-3/2): use recurrence g[k] = (1/(k*s[0])) * sum ... */
        if (k == 0) {
            g1c[0] = std::pow(s1c[0], -1.5);
            h1c[0] = std::pow(s1c[0], -2.5);
            e1c[0] = std::pow(s1c[0], -3.5);
            f1c[0] = std::pow(s1c[0], -4.5);
            g2c[0] = std::pow(s2c[0], -1.5);
            h2c[0] = std::pow(s2c[0], -2.5);
            e2c[0] = std::pow(s2c[0], -3.5);
            f2c[0] = std::pow(s2c[0], -4.5);
        } else {
            /* Recurrence for g = s^alpha:
             *   g[k] = (1/(k * s[0])) sum_{j=0}^{k-1} (alpha*(k-j) - j) * s[k-j] * g[j]
             */
            auto power_coeff = [](const TaylorScalar& sc, TaylorScalar& gc,
                                  double alpha, int kk) {
                double inv0 = 1.0 / sc[0];
                double s = 0.0;
                for (int j = 0; j < kk; ++j)
                    s += (alpha * (kk - j) - j) * sc[kk - j] * gc[j];
                gc[kk] = inv0 * s / (double)kk;
            };

            power_coeff(s1c, g1c, -1.5, k);
            power_coeff(s1c, h1c, -2.5, k);
            power_coeff(s1c, e1c, -3.5, k);
            power_coeff(s1c, f1c, -4.5, k);
            power_coeff(s2c, g2c, -1.5, k);
            power_coeff(s2c, h2c, -2.5, k);
            power_coeff(s2c, e2c, -3.5, k);
            power_coeff(s2c, f2c, -4.5, k);
        }
    };

    auto build_products_at_k = [&](int k) {
        /* q1*g1 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += q1c[j] * g1c[k-j];
          q1g1[k] = s; }
        /* q2*g2 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += q2c[j] * g2c[k-j];
          q2g2[k] = s; }
        /* y*g1 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += yc[j] * g1c[k-j];
          yg1[k] = s; }
        /* y*g2 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += yc[j] * g2c[k-j];
          yg2[k] = s; }
        /* z*g1 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += zc[j] * g1c[k-j];
          zg1[k] = s; }
        /* z*g2 */
        { double s = 0.0;
          for (int j = 0; j <= k; ++j) s += zc[j] * g2c[k-j];
          zg2[k] = s; }
    };

    /* Build k=0 intermediates and state */
    build_intermediates_at_k(0);
    build_powers_at_k(0);
    build_products_at_k(0);

    /* Now iterate: for k = 0..K-1, compute acceleration[k], advance state to k+1 */
    for (int k = 0; k < K; ++k) {
        /* Acceleration Taylor coefficients at order k:
         *   ax = 2*vy + x - (1-mu)*q1*g1 - mu*q2*g2
         *   ay = -2*vx + y - (1-mu)*y*g1 - mu*y*g2
         *   az = -(1-mu)*z*g1 - mu*z*g2
         *
         * In Taylor-coefficient form these are just evaluated at index k.
         */
        double axk = 2.0*vyc[k] + xc[k] - mu1*q1g1[k] - mu*q2g2[k];
        double ayk = -2.0*vxc[k] + yc[k] - mu1*yg1[k] - mu*yg2[k];
        double azk = -mu1*zg1[k] - mu*zg2[k];

        /* Advance state to order k+1 */
        double inv_kp1 = 1.0 / (double)(k + 1);
        xc[k+1]  = vxc[k] * inv_kp1;
        yc[k+1]  = vyc[k] * inv_kp1;
        zc[k+1]  = vzc[k] * inv_kp1;
        vxc[k+1] = axk * inv_kp1;
        vyc[k+1] = ayk * inv_kp1;
        vzc[k+1] = azk * inv_kp1;

        /* Build intermediates at order k+1 for the NEXT iteration */
        if (k + 1 <= K) {
            build_intermediates_at_k(k + 1);
            build_powers_at_k(k + 1);
            build_products_at_k(k + 1);
        }
    }

    /*==================================================================
     * STEP 2: Build Taylor coefficient series for the spatial
     * derivative tensors A(t), B(t), C(t), D(t) of the CR3BP
     * vector field, evaluated along the Taylor-expanded trajectory.
     *
     * We need these as Taylor-coefficient arrays to perform the
     * Cauchy-product contractions in the variational equations.
     *
     * A(i,p)         6x6     = Jacobian of f
     * B(i,p,q)       6x6x6   = Hessian of f
     * C(i,p,q,r)     6x6x6x6 = 3rd spatial derivative of f
     * D(i,p,q,r,s)   6x6x6x6x6 = 4th spatial derivative of f
     *
     * The velocity rows (i=0,1,2) have trivial entries:
     *   A(0,3) = 1, A(1,4) = 1, A(2,5) = 1 (constant)
     *   B = C = D = 0 for i < 3.
     *
     * The acceleration rows (i=3,4,5) involve the gravitational
     * potential derivatives. We derive them below.
     *==================================================================*/

    /* Allocate Taylor-coefficient storage for A, B, C, D.
     * Ac[k] is stored at Ac_data + k*SZ1, etc. */
    std::vector<double> Ac_data(SZ1 * Kp1, 0.0);
    std::vector<double> Bc_data(SZ2 * Kp1, 0.0);
    std::vector<double> Cc_data(SZ3 * Kp1, 0.0);
    std::vector<double> Dc_data(SZ4 * Kp1, 0.0);

    double* Ac = Ac_data.data();
    double* Bc = Bc_data.data();
    double* Cc = Cc_data.data();
    double* Dc = Dc_data.data();

    /*------------------------------------------------------------------
     * Precompute all needed product Taylor series for building A,B,C,D.
     *
     * We need many products of coordinate components with various
     * inverse-power scalars. Let's build them all here.
     *
     * Notation:
     *   coord = {q1, q2, y, z}    (indices relative to body 1 and 2)
     *   power = {g, h, e, f}      (s^{-3/2}, s^{-5/2}, s^{-7/2}, s^{-9/2})
     *
     * For the Jacobian (A), we need products of two coordinates with h:
     *   q1*q1*h1, q1*y*h1, q1*z*h1, y*y*h1, y*z*h1, z*z*h1
     *   (same for body 2)
     *
     * For the Hessian (B), we need products of three coordinates with e:
     *   e.g., q1*q1*q1*e1, q1*q1*y*e1, etc.
     *
     * For C, we need products of four coordinates with f.
     *
     * We'll build these using iterative Cauchy products.
     *------------------------------------------------------------------*/

    /* Helper to compute Cauchy product of two series a,b and store in c,
     * all of length Kp1. */
    auto ts_mul = [&](const TaylorScalar& a, const TaylorScalar& b, TaylorScalar& c) {
        for (int k = 0; k <= K; ++k) {
            double s = 0.0;
            for (int j = 0; j <= k; ++j) s += a[j] * b[k-j];
            c[k] = s;
        }
    };

    /* ---- Products of coordinates with h (s^{-5/2}) for body 1 ---- */
    TaylorScalar q1h1(K), yh1(K), zh1(K);
    TaylorScalar q1h2_ts(K), yh2_ts(K), zh2_ts(K); /* "q2*h2" etc, named to avoid confusion */
    ts_mul(q1c, h1c, q1h1);
    ts_mul(yc,  h1c, yh1);
    ts_mul(zc,  h1c, zh1);
    /* body 2 */
    TaylorScalar q2h2(K);
    ts_mul(q2c, h2c, q2h2);
    ts_mul(yc,  h2c, yh2_ts);
    ts_mul(zc,  h2c, zh2_ts);

    /* ---- Two-coordinate products with h for A ---- */
    TaylorScalar q1q1h1(K), q1yh1(K), q1zh1(K), yyh1(K), yzh1(K), zzh1(K);
    TaylorScalar q2q2h2(K), q2yh2(K), q2zh2(K), yyh2(K), yzh2(K), zzh2(K);
    ts_mul(q1c, q1h1, q1q1h1);
    ts_mul(yc,  q1h1, q1yh1);  /* = q1*y*h1 */
    ts_mul(zc,  q1h1, q1zh1);
    ts_mul(yc,  yh1,  yyh1);
    ts_mul(zc,  yh1,  yzh1);
    ts_mul(zc,  zh1,  zzh1);
    ts_mul(q2c, q2h2, q2q2h2);
    ts_mul(yc,  q2h2, q2yh2);
    ts_mul(zc,  q2h2, q2zh2);
    ts_mul(yc,  yh2_ts, yyh2);
    ts_mul(zc,  yh2_ts, yzh2);
    ts_mul(zc,  zh2_ts, zzh2);

    /* ---- Products with e (s^{-7/2}) for B ---- */
    TaylorScalar q1e1(K), ye1(K), ze1(K);
    TaylorScalar q2e2(K), ye2_ts(K), ze2_ts(K);
    ts_mul(q1c, e1c, q1e1);
    ts_mul(yc,  e1c, ye1);
    ts_mul(zc,  e1c, ze1);
    ts_mul(q2c, e2c, q2e2);
    ts_mul(yc,  e2c, ye2_ts);
    ts_mul(zc,  e2c, ze2_ts);

    /* Two-coordinate products with e for B */
    TaylorScalar q1q1e1(K), q1ye1(K), q1ze1(K), yye1(K), yze1(K), zze1(K);
    TaylorScalar q2q2e2(K), q2ye2(K), q2ze2(K), yye2(K), yze2(K), zze2(K);
    ts_mul(q1c, q1e1, q1q1e1);
    ts_mul(yc,  q1e1, q1ye1);
    ts_mul(zc,  q1e1, q1ze1);
    ts_mul(yc,  ye1,  yye1);
    ts_mul(zc,  ye1,  yze1);
    ts_mul(zc,  ze1,  zze1);
    ts_mul(q2c, q2e2, q2q2e2);
    ts_mul(yc,  q2e2, q2ye2);
    ts_mul(zc,  q2e2, q2ze2);
    ts_mul(yc,  ye2_ts, yye2);
    ts_mul(zc,  ye2_ts, yze2);
    ts_mul(zc,  ze2_ts, zze2);

    /* Three-coordinate products with e for B (needed for certain B components) */
    TaylorScalar q1q1q1e1(K), q1q1ye1(K), q1q1ze1(K);
    TaylorScalar q1yye1(K), q1yze1(K), q1zze1(K);
    TaylorScalar yyye1(K), yyze1(K), yzze1(K), zzze1(K);
    /* body 2 */
    TaylorScalar q2q2q2e2(K), q2q2ye2(K), q2q2ze2(K);
    TaylorScalar q2yye2(K), q2yze2(K), q2zze2(K);
    TaylorScalar yyye2(K), yyze2(K), yzze2(K), zzze2(K);

    ts_mul(q1c, q1q1e1, q1q1q1e1);
    ts_mul(yc,  q1q1e1, q1q1ye1);
    ts_mul(zc,  q1q1e1, q1q1ze1);
    ts_mul(yc,  q1ye1,  q1yye1);
    ts_mul(zc,  q1ye1,  q1yze1);
    ts_mul(zc,  q1ze1,  q1zze1);
    ts_mul(yc,  yye1,   yyye1);
    ts_mul(zc,  yye1,   yyze1);
    ts_mul(zc,  yze1,   yzze1);
    ts_mul(zc,  zze1,   zzze1);

    ts_mul(q2c, q2q2e2, q2q2q2e2);
    ts_mul(yc,  q2q2e2, q2q2ye2);
    ts_mul(zc,  q2q2e2, q2q2ze2);
    ts_mul(yc,  q2ye2,  q2yye2);
    ts_mul(zc,  q2ye2,  q2yze2);
    ts_mul(zc,  q2ze2,  q2zze2);
    ts_mul(yc,  yye2,   yyye2);
    ts_mul(zc,  yye2,   yyze2);
    ts_mul(zc,  yze2,   yzze2);
    ts_mul(zc,  zze2,   zzze2);

    /* ---- Products with f (s^{-9/2}) for D ---- */
    TaylorScalar q1f1(K), yf1(K), zf1(K);
    TaylorScalar q2f2(K), yf2_ts(K), zf2_ts(K);
    ts_mul(q1c, f1c, q1f1);
    ts_mul(yc,  f1c, yf1);
    ts_mul(zc,  f1c, zf1);
    ts_mul(q2c, f2c, q2f2);
    ts_mul(yc,  f2c, yf2_ts);
    ts_mul(zc,  f2c, zf2_ts);

    /* Two-coordinate products with f */
    TaylorScalar q1q1f1(K), q1yf1(K), q1zf1(K), yyf1(K), yzf1(K), zzf1(K);
    TaylorScalar q2q2f2(K), q2yf2(K), q2zf2(K), yyf2(K), yzf2(K), zzf2(K);
    ts_mul(q1c, q1f1, q1q1f1);
    ts_mul(yc,  q1f1, q1yf1);
    ts_mul(zc,  q1f1, q1zf1);
    ts_mul(yc,  yf1,  yyf1);
    ts_mul(zc,  yf1,  yzf1);
    ts_mul(zc,  zf1,  zzf1);
    ts_mul(q2c, q2f2, q2q2f2);
    ts_mul(yc,  q2f2, q2yf2);
    ts_mul(zc,  q2f2, q2zf2);
    ts_mul(yc,  yf2_ts, yyf2);
    ts_mul(zc,  yf2_ts, yzf2);
    ts_mul(zc,  zf2_ts, zzf2);

    /* Three-coordinate products with f for C */
    TaylorScalar q1q1q1f1(K), q1q1yf1(K), q1q1zf1(K);
    TaylorScalar q1yyf1(K), q1yzf1(K), q1zzf1(K);
    TaylorScalar yyyf1(K), yyzf1(K), yzzf1(K), zzzf1(K);
    TaylorScalar q2q2q2f2(K), q2q2yf2(K), q2q2zf2(K);
    TaylorScalar q2yyf2(K), q2yzf2(K), q2zzf2(K);
    TaylorScalar yyyf2(K), yyzf2(K), yzzf2(K), zzzf2(K);

    ts_mul(q1c, q1q1f1, q1q1q1f1);
    ts_mul(yc,  q1q1f1, q1q1yf1);
    ts_mul(zc,  q1q1f1, q1q1zf1);
    ts_mul(yc,  q1yf1,  q1yyf1);
    ts_mul(zc,  q1yf1,  q1yzf1);
    ts_mul(zc,  q1zf1,  q1zzf1);
    ts_mul(yc,  yyf1,   yyyf1);
    ts_mul(zc,  yyf1,   yyzf1);
    ts_mul(zc,  yzf1,   yzzf1);
    ts_mul(zc,  zzf1,   zzzf1);

    ts_mul(q2c, q2q2f2, q2q2q2f2);
    ts_mul(yc,  q2q2f2, q2q2yf2);
    ts_mul(zc,  q2q2f2, q2q2zf2);
    ts_mul(yc,  q2yf2,  q2yyf2);
    ts_mul(zc,  q2yf2,  q2yzf2);
    ts_mul(zc,  q2zf2,  q2zzf2);
    ts_mul(yc,  yyf2,   yyyf2);
    ts_mul(zc,  yyf2,   yyzf2);
    ts_mul(zc,  yzf2,   yzzf2);
    ts_mul(zc,  zzf2,   zzzf2);

    /* Four-coordinate products with f for D */
    TaylorScalar q1q1q1q1f1(K), q1q1q1yf1(K), q1q1q1zf1(K);
    TaylorScalar q1q1yyf1(K), q1q1yzf1(K), q1q1zzf1(K);
    TaylorScalar q1yyyf1(K), q1yyzf1(K), q1yzzf1(K), q1zzzf1(K);
    TaylorScalar yyyyf1(K), yyyzf1(K), yyzzf1(K), yzzzf1(K), zzzzf1(K);

    TaylorScalar q2q2q2q2f2(K), q2q2q2yf2(K), q2q2q2zf2(K);
    TaylorScalar q2q2yyf2(K), q2q2yzf2(K), q2q2zzf2(K);
    TaylorScalar q2yyyf2(K), q2yyzf2(K), q2yzzf2(K), q2zzzf2(K);
    TaylorScalar yyyyf2(K), yyyzf2(K), yyzzf2(K), yzzzf2(K), zzzzf2(K);

    ts_mul(q1c, q1q1q1f1, q1q1q1q1f1);
    ts_mul(yc,  q1q1q1f1, q1q1q1yf1);
    ts_mul(zc,  q1q1q1f1, q1q1q1zf1);
    ts_mul(yc,  q1q1yf1,  q1q1yyf1);
    ts_mul(zc,  q1q1yf1,  q1q1yzf1);
    ts_mul(zc,  q1q1zf1,  q1q1zzf1);
    ts_mul(yc,  q1yyf1,   q1yyyf1);
    ts_mul(zc,  q1yyf1,   q1yyzf1);
    ts_mul(zc,  q1yzf1,   q1yzzf1);
    ts_mul(zc,  q1zzf1,   q1zzzf1);
    ts_mul(yc,  yyyf1,    yyyyf1);
    ts_mul(zc,  yyyf1,    yyyzf1);
    ts_mul(zc,  yyzf1,    yyzzf1);
    ts_mul(zc,  yzzf1,    yzzzf1);
    ts_mul(zc,  zzzf1,    zzzzf1);

    ts_mul(q2c, q2q2q2f2, q2q2q2q2f2);
    ts_mul(yc,  q2q2q2f2, q2q2q2yf2);
    ts_mul(zc,  q2q2q2f2, q2q2q2zf2);
    ts_mul(yc,  q2q2yf2,  q2q2yyf2);
    ts_mul(zc,  q2q2yf2,  q2q2yzf2);
    ts_mul(zc,  q2q2zf2,  q2q2zzf2);
    ts_mul(yc,  q2yyf2,   q2yyyf2);
    ts_mul(zc,  q2yyf2,   q2yyzf2);
    ts_mul(zc,  q2yzf2,   q2yzzf2);
    ts_mul(zc,  q2zzf2,   q2zzzf2);
    ts_mul(yc,  yyyf2,    yyyyf2);
    ts_mul(zc,  yyyf2,    yyyzf2);
    ts_mul(zc,  yyzf2,    yyzzf2);
    ts_mul(zc,  yzzf2,    yzzzf2);
    ts_mul(zc,  zzzf2,    zzzzf2);

    /*------------------------------------------------------------------
     * Now build A(t), B(t), C(t), D(t) Taylor coefficient arrays.
     *
     * CR3BP gravitational potential gradient w.r.t. position:
     *   Ux  = -(1-mu)*(x+mu)/r1^3 - mu*(x-1+mu)/r2^3  + x
     *   Uy  = -(1-mu)*y/r1^3      - mu*y/r2^3          + y
     *   Uz  = -(1-mu)*z/r1^3      - mu*z/r2^3
     *
     * The CR3BP acceleration (vector field rows i=3,4,5):
     *   f_3 = 2*vy + Ux
     *   f_4 = -2*vx + Uy
     *   f_5 = Uz
     *
     * Index convention for the state vector:
     *   0=x, 1=y, 2=z, 3=vx, 4=vy, 5=vz
     *
     * ---- JACOBIAN  A(i,p) = ∂f_i/∂X_p ----
     *
     * For i=0,1,2 (velocity components):
     *   A(0,3) = 1, A(1,4) = 1, A(2,5) = 1  (constant)
     *
     * For i=3: f_3 = 2*vy + x - mu1*q1*g1 - mu*q2*g2
     *   ∂f_3/∂x  = 1 - mu1*g1 + 3*mu1*q1^2*h1 - mu*g2 + 3*mu*q2^2*h2
     *   ∂f_3/∂y  = 3*mu1*q1*y*h1 + 3*mu*q2*y*h2
     *   ∂f_3/∂z  = 3*mu1*q1*z*h1 + 3*mu*q2*z*h2
     *   ∂f_3/∂vx = 0
     *   ∂f_3/∂vy = 2
     *   ∂f_3/∂vz = 0
     *
     * For i=4: f_4 = -2*vx + y - mu1*y*g1 - mu*y*g2
     *   ∂f_4/∂x  = 3*mu1*q1*y*h1 + 3*mu*q2*y*h2
     *   ∂f_4/∂y  = 1 - mu1*g1 + 3*mu1*y^2*h1 - mu*g2 + 3*mu*y^2*h2
     *   ∂f_4/∂z  = 3*mu1*y*z*h1 + 3*mu*y*z*h2
     *   ∂f_4/∂vx = -2
     *   ∂f_4/∂vy = 0
     *   ∂f_4/∂vz = 0
     *
     * For i=5: f_5 = -mu1*z*g1 - mu*z*g2
     *   ∂f_5/∂x  = 3*mu1*q1*z*h1 + 3*mu*q2*z*h2
     *   ∂f_5/∂y  = 3*mu1*y*z*h1 + 3*mu*y*z*h2
     *   ∂f_5/∂z  = -mu1*g1 + 3*mu1*z^2*h1 - mu*g2 + 3*mu*z^2*h2
     *   ∂f_5/∂vx = 0
     *   ∂f_5/∂vy = 0
     *   ∂f_5/∂vz = 0
     *------------------------------------------------------------------*/

    /* Fill A Taylor coefficients for each order k */
    for (int k = 0; k <= K; ++k) {
        double* Ak = Ac + k * SZ1;  /* pointer to A[k] block */

        /* velocity rows: constant, only k=0 */
        if (k == 0) {
            Ak[idx2(0,3)] = 1.0;
            Ak[idx2(1,4)] = 1.0;
            Ak[idx2(2,5)] = 1.0;
        }

        /* Acceleration rows */
        /* A(3,0) = ∂f_3/∂x = 1(k==0) - mu1*g1 + 3*mu1*q1^2*h1 - mu*g2 + 3*mu*q2^2*h2 */
        Ak[idx2(3,0)] = (k==0 ? 1.0 : 0.0) - mu1*g1c[k] + 3.0*mu1*q1q1h1[k]
                         - mu*g2c[k] + 3.0*mu*q2q2h2[k];

        /* A(3,1) = 3*mu1*q1*y*h1 + 3*mu*q2*y*h2 */
        Ak[idx2(3,1)] = 3.0*mu1*q1yh1[k] + 3.0*mu*q2yh2[k];

        /* A(3,2) = 3*mu1*q1*z*h1 + 3*mu*q2*z*h2 */
        Ak[idx2(3,2)] = 3.0*mu1*q1zh1[k] + 3.0*mu*q2zh2[k];

        /* A(3,3) = 0 */
        /* A(3,4) = 2 (constant) */
        if (k == 0) Ak[idx2(3,4)] = 2.0;
        /* A(3,5) = 0 */

        /* A(4,0) = A(3,1) by symmetry of the gravitational part */
        Ak[idx2(4,0)] = Ak[idx2(3,1)];

        /* A(4,1) = 1(k==0) - mu1*g1 + 3*mu1*y^2*h1 - mu*g2 + 3*mu*y^2*h2 */
        Ak[idx2(4,1)] = (k==0 ? 1.0 : 0.0) - mu1*g1c[k] + 3.0*mu1*yyh1[k]
                         - mu*g2c[k] + 3.0*mu*yyh2[k];

        /* A(4,2) = 3*mu1*y*z*h1 + 3*mu*y*z*h2 */
        Ak[idx2(4,2)] = 3.0*mu1*yzh1[k] + 3.0*mu*yzh2[k];

        /* A(4,3) = -2 (constant) */
        if (k == 0) Ak[idx2(4,3)] = -2.0;

        /* A(5,0) = A(3,2) by symmetry */
        Ak[idx2(5,0)] = Ak[idx2(3,2)];

        /* A(5,1) = A(4,2) by symmetry */
        Ak[idx2(5,1)] = Ak[idx2(4,2)];

        /* A(5,2) = -mu1*g1 + 3*mu1*z^2*h1 - mu*g2 + 3*mu*z^2*h2 */
        Ak[idx2(5,2)] = -mu1*g1c[k] + 3.0*mu1*zzh1[k]
                         - mu*g2c[k] + 3.0*mu*zzh2[k];
    }

    /*------------------------------------------------------------------
     * HESSIAN  B(i,p,q) = ∂²f_i / (∂X_p ∂X_q)
     *
     * Only rows i=3,4,5 (acceleration) are nonzero.
     * Only columns p,q in {0,1,2} (position) are nonzero.
     *
     * Computed from derivatives of the Jacobian entries w.r.t. position.
     *
     * Define the gravitational contribution from body j:
     *   U_j = -m_j / r_j   (potential)
     *
     * The key building blocks are:
     *   ∂(c_p * g_j)/∂c_q = delta(p,q)*g_j - 3*c_p*c_q*h_j
     *   ∂(c_p*c_q*h_j)/∂c_r = (delta(p,r)*c_q + delta(q,r)*c_p)*h_j - 5*c_p*c_q*c_r*e_j
     *
     * where c_0 = q_j, c_1 = y, c_2 = z for body j.
     *
     * For each body j, the Hessian contribution to f_i is:
     *   B_j(i,p,q) = ∂A_j(i,p) / ∂c_q
     *
     * Let's work through the Hessian systematically.
     * The gravitational part of A(3+α, β) for body j is:
     *
     *   G_j(α,β) = -m_j * δ(α,β) * g_j + 3 * m_j * c_α * c_β * h_j
     *
     * where α,β ∈ {0,1,2} and c_0=q_j, c_1=y, c_2=z.
     *
     * Then:
     *   B_j(3+α,β,γ) = ∂G_j(α,β)/∂c_γ
     *     = -m_j * δ(α,β) * ∂g_j/∂c_γ + 3*m_j * (δ(α,γ)*c_β + c_α*δ(β,γ))*h_j
     *       + 3*m_j*c_α*c_β * ∂h_j/∂c_γ
     *
     *   ∂g_j/∂c_γ = -3 * c_γ * h_j
     *   ∂h_j/∂c_γ = -5 * c_γ * e_j
     *
     * So:
     *   B_j(3+α,β,γ) = 3*m_j*δ(α,β)*c_γ*h_j
     *                   + 3*m_j*(δ(α,γ)*c_β + δ(β,γ)*c_α)*h_j
     *                   - 15*m_j*c_α*c_β*c_γ*e_j
     *
     * Total: B(3+α,β,γ) = B_1(3+α,β,γ) + B_2(3+α,β,γ)
     *   where m_1 = mu1 = 1-mu,  c = {q1,y,z} for body 1
     *         m_2 = mu,           c = {q2,y,z} for body 2
     *
     * B(i,p,q) = 0 for i<3 or p>=3 or q>=3.
     *------------------------------------------------------------------*/

    /* Helper: get the Taylor coefficient of coordinate c_idx for body b at order k.
     * c_idx: 0 -> q (q1 or q2), 1 -> y, 2 -> z
     * Returns pointer to the TaylorScalar's data. */

    /* We'll build a table of the 3-coordinate product with e for each combination.
     * coord_e[b][α][β][γ] at order k = c_α * c_β * c_γ * e
     * coord_h[b][α][β] at order k = c_α * c_β * h
     * coord_h1[b][α] at order k = c_α * h
     *
     * We already have these precomputed. Let's index them properly. */

    /* We need the Kronecker delta: δ(α,γ) etc. */

    /* Build B for each order k.
     * We use the formula:
     *   B_j(3+α,β,γ)[k] = 3*m_j*(δ(α,β)*c_γ*h_j + δ(α,γ)*c_β*h_j + δ(β,γ)*c_α*h_j)[k]
     *                     - 15*m_j*(c_α*c_β*c_γ*e_j)[k]
     */

    /* Helper arrays for coordinate*h and coordinate*coordinate*coordinate*e products.
     * We index by body and coordinate indices.
     * coord_h_ts[body][coord_idx] is a pointer to the Taylor scalar for c_coord * h_body.
     * coord3_e_ts[body][a][b][c] is c_a*c_b*c_c*e (but we need to be clever about storage).
     */

    /* Rather than building a general indexing scheme, let's use lookup functions. */

    /* For body 1: c[0] = q1, c[1] = y, c[2] = z
     * For body 2: c[0] = q2, c[1] = y, c[2] = z */

    /* c_α * h at order k for body j */
    auto get_ch = [&](int body, int a, int k) -> double {
        if (body == 0) { /* body 1 */
            if (a == 0) return q1h1[k];
            if (a == 1) return yh1[k];
            return zh1[k]; /* a == 2 */
        } else { /* body 2 */
            if (a == 0) return q2h2[k];
            if (a == 1) return yh2_ts[k];
            return zh2_ts[k];
        }
    };

    /* c_α * c_β * c_γ * e at order k for body j (symmetric in all indices) */
    auto get_ccce = [&](int body, int a, int b, int c, int k) -> double {
        /* sort */
        if (a > b) std::swap(a, b);
        if (b > c) std::swap(b, c);
        if (a > b) std::swap(a, b);
        if (body == 0) {
            if (a==0&&b==0&&c==0) return q1q1q1e1[k];
            if (a==0&&b==0&&c==1) return q1q1ye1[k];
            if (a==0&&b==0&&c==2) return q1q1ze1[k];
            if (a==0&&b==1&&c==1) return q1yye1[k];
            if (a==0&&b==1&&c==2) return q1yze1[k];
            if (a==0&&b==2&&c==2) return q1zze1[k];
            if (a==1&&b==1&&c==1) return yyye1[k];
            if (a==1&&b==1&&c==2) return yyze1[k];
            if (a==1&&b==2&&c==2) return yzze1[k];
            return zzze1[k];
        } else {
            if (a==0&&b==0&&c==0) return q2q2q2e2[k];
            if (a==0&&b==0&&c==1) return q2q2ye2[k];
            if (a==0&&b==0&&c==2) return q2q2ze2[k];
            if (a==0&&b==1&&c==1) return q2yye2[k];
            if (a==0&&b==1&&c==2) return q2yze2[k];
            if (a==0&&b==2&&c==2) return q2zze2[k];
            if (a==1&&b==1&&c==1) return yyye2[k];
            if (a==1&&b==1&&c==2) return yyze2[k];
            if (a==1&&b==2&&c==2) return yzze2[k];
            return zzze2[k];
        }
    };

    /* c_α * c_β * c_γ * c_δ * f at order k for body j */
    auto get_ccccf = [&](int body, int a, int b, int c, int d, int k) -> double {
        int idx[4] = {a, b, c, d};
        std::sort(idx, idx + 4);
        a = idx[0]; b = idx[1]; c = idx[2]; d = idx[3];
        if (body == 0) {
            if (a==0&&b==0&&c==0&&d==0) return q1q1q1q1f1[k];
            if (a==0&&b==0&&c==0&&d==1) return q1q1q1yf1[k];
            if (a==0&&b==0&&c==0&&d==2) return q1q1q1zf1[k];
            if (a==0&&b==0&&c==1&&d==1) return q1q1yyf1[k];
            if (a==0&&b==0&&c==1&&d==2) return q1q1yzf1[k];
            if (a==0&&b==0&&c==2&&d==2) return q1q1zzf1[k];
            if (a==0&&b==1&&c==1&&d==1) return q1yyyf1[k];
            if (a==0&&b==1&&c==1&&d==2) return q1yyzf1[k];
            if (a==0&&b==1&&c==2&&d==2) return q1yzzf1[k];
            if (a==0&&b==2&&c==2&&d==2) return q1zzzf1[k];
            if (a==1&&b==1&&c==1&&d==1) return yyyyf1[k];
            if (a==1&&b==1&&c==1&&d==2) return yyyzf1[k];
            if (a==1&&b==1&&c==2&&d==2) return yyzzf1[k];
            if (a==1&&b==2&&c==2&&d==2) return yzzzf1[k];
            return zzzzf1[k];
        } else {
            if (a==0&&b==0&&c==0&&d==0) return q2q2q2q2f2[k];
            if (a==0&&b==0&&c==0&&d==1) return q2q2q2yf2[k];
            if (a==0&&b==0&&c==0&&d==2) return q2q2q2zf2[k];
            if (a==0&&b==0&&c==1&&d==1) return q2q2yyf2[k];
            if (a==0&&b==0&&c==1&&d==2) return q2q2yzf2[k];
            if (a==0&&b==0&&c==2&&d==2) return q2q2zzf2[k];
            if (a==0&&b==1&&c==1&&d==1) return q2yyyf2[k];
            if (a==0&&b==1&&c==1&&d==2) return q2yyzf2[k];
            if (a==0&&b==1&&c==2&&d==2) return q2yzzf2[k];
            if (a==0&&b==2&&c==2&&d==2) return q2zzzf2[k];
            if (a==1&&b==1&&c==1&&d==1) return yyyyf2[k];
            if (a==1&&b==1&&c==1&&d==2) return yyyzf2[k];
            if (a==1&&b==1&&c==2&&d==2) return yyzzf2[k];
            if (a==1&&b==2&&c==2&&d==2) return yzzzf2[k];
            return zzzzf2[k];
        }
    };

    /* c_α * e at order k */
    auto get_ce = [&](int body, int a, int k) -> double {
        if (body == 0) {
            if (a == 0) return q1e1[k];
            if (a == 1) return ye1[k];
            return ze1[k];
        } else {
            if (a == 0) return q2e2[k];
            if (a == 1) return ye2_ts[k];
            return ze2_ts[k];
        }
    };

    /* c_α * c_β * e at order k */
    auto get_cce = [&](int body, int a, int b, int k) -> double {
        if (a > b) std::swap(a, b);
        if (body == 0) {
            if (a==0&&b==0) return q1q1e1[k];
            if (a==0&&b==1) return q1ye1[k];
            if (a==0&&b==2) return q1ze1[k];
            if (a==1&&b==1) return yye1[k];
            if (a==1&&b==2) return yze1[k];
            return zze1[k];
        } else {
            if (a==0&&b==0) return q2q2e2[k];
            if (a==0&&b==1) return q2ye2[k];
            if (a==0&&b==2) return q2ze2[k];
            if (a==1&&b==1) return yye2[k];
            if (a==1&&b==2) return yze2[k];
            return zze2[k];
        }
    };

    /* c_α * c_β * c_γ * f at order k */
    auto get_cccf = [&](int body, int a, int b, int c, int k) -> double {
        int idx[3] = {a, b, c};
        std::sort(idx, idx + 3);
        a = idx[0]; b = idx[1]; c = idx[2];
        if (body == 0) {
            if (a==0&&b==0&&c==0) return q1q1q1f1[k];
            if (a==0&&b==0&&c==1) return q1q1yf1[k];
            if (a==0&&b==0&&c==2) return q1q1zf1[k];
            if (a==0&&b==1&&c==1) return q1yyf1[k];
            if (a==0&&b==1&&c==2) return q1yzf1[k];
            if (a==0&&b==2&&c==2) return q1zzf1[k];
            if (a==1&&b==1&&c==1) return yyyf1[k];
            if (a==1&&b==1&&c==2) return yyzf1[k];
            if (a==1&&b==2&&c==2) return yzzf1[k];
            return zzzf1[k];
        } else {
            if (a==0&&b==0&&c==0) return q2q2q2f2[k];
            if (a==0&&b==0&&c==1) return q2q2yf2[k];
            if (a==0&&b==0&&c==2) return q2q2zf2[k];
            if (a==0&&b==1&&c==1) return q2yyf2[k];
            if (a==0&&b==1&&c==2) return q2yzf2[k];
            if (a==0&&b==2&&c==2) return q2zzf2[k];
            if (a==1&&b==1&&c==1) return yyyf2[k];
            if (a==1&&b==1&&c==2) return yyzf2[k];
            if (a==1&&b==2&&c==2) return yzzf2[k];
            return zzzf2[k];
        }
    };

    auto dkr = [](int a, int b) -> double { return (a == b) ? 1.0 : 0.0; };

    /* Fill B for each order k */
    for (int k = 0; k <= K; ++k) {
        double* Bk = Bc + k * SZ2;

        for (int alpha = 0; alpha < 3; ++alpha) {
            int i = 3 + alpha;  /* row of f */
            for (int beta = 0; beta < 3; ++beta) {
                for (int gamma = 0; gamma < 3; ++gamma) {
                    double val = 0.0;
                    /* Body 1 contribution */
                    {
                        double mj = mu1;
                        /* 3*mj*(δ(α,β)*c_γ*h + δ(α,γ)*c_β*h + δ(β,γ)*c_α*h) */
                        val += 3.0*mj*(dkr(alpha,beta)*get_ch(0,gamma,k)
                                     + dkr(alpha,gamma)*get_ch(0,beta,k)
                                     + dkr(beta,gamma)*get_ch(0,alpha,k));
                        /* -15*mj*c_α*c_β*c_γ*e */
                        val -= 15.0*mj*get_ccce(0,alpha,beta,gamma,k);
                    }
                    /* Body 2 contribution */
                    {
                        double mj = mu;
                        val += 3.0*mj*(dkr(alpha,beta)*get_ch(1,gamma,k)
                                     + dkr(alpha,gamma)*get_ch(1,beta,k)
                                     + dkr(beta,gamma)*get_ch(1,alpha,k));
                        val -= 15.0*mj*get_ccce(1,alpha,beta,gamma,k);
                    }
                    Bk[idx3(i, beta, gamma)] = val;
                }
            }
        }
    }

    /*------------------------------------------------------------------
     * THIRD SPATIAL DERIVATIVE  C(i,p,q,r) = ∂³f_i / (∂X_p ∂X_q ∂X_r)
     *
     * Differentiating B:
     *   C_j(3+α,β,γ,δ) = ∂B_j(3+α,β,γ)/∂c_δ
     *
     *   B_j(3+α,β,γ) = 3*m_j*(δ(α,β)*c_γ + δ(α,γ)*c_β + δ(β,γ)*c_α)*h_j
     *                 - 15*m_j*c_α*c_β*c_γ*e_j
     *
     *   ∂/∂c_δ [c_x * h_j] = δ(x,δ)*h_j + c_x * (-5*c_δ*e_j)
     *                       = δ(x,δ)*h_j - 5*c_x*c_δ*e_j
     *
     *   ∂/∂c_δ [c_α*c_β*c_γ*e_j]
     *     = (δ(α,δ)*c_β*c_γ + δ(β,δ)*c_α*c_γ + δ(γ,δ)*c_α*c_β)*e_j
     *       + c_α*c_β*c_γ*(-7*c_δ*f_j)
     *
     * So:
     *   C_j(3+α,β,γ,δ) =
     *     3*m_j*{
     *       δ(α,β)*[δ(γ,δ)*h_j - 5*c_γ*c_δ*e_j]
     *     + δ(α,γ)*[δ(β,δ)*h_j - 5*c_β*c_δ*e_j]
     *     + δ(β,γ)*[δ(α,δ)*h_j - 5*c_α*c_δ*e_j]
     *     }
     *   - 15*m_j*{
     *       (δ(α,δ)*c_β*c_γ + δ(β,δ)*c_α*c_γ + δ(γ,δ)*c_α*c_β)*e_j
     *     - 7*c_α*c_β*c_γ*c_δ*f_j
     *     }
     *
     * Simplifying:
     *   C_j = 3*m_j*h_j*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)
     *       - 15*m_j*e_j*(δαβ*cc(γ,δ) + δαγ*cc(β,δ) + δαδ*cc(β,γ)
     *                    + δβγ*cc(α,δ) + δβδ*cc(α,γ) + δγδ*cc(α,β))
     *       + 105*m_j*f_j*cc(α,β,γ,δ)
     *
     * Wait, let me redo this more carefully. The e terms:
     *   From the h-derivative: -5*m_j*3*(δαβ*c_γ*c_δ + δαγ*c_β*c_δ + δβγ*c_α*c_δ)*e_j
     *   From the triple product: -15*m_j*(δαδ*c_β*c_γ + δβδ*c_α*c_γ + δγδ*c_α*c_β)*e_j
     *
     * Total e coefficient: -15*m_j*e_j*(δαβ*cc(γ,δ) + δαγ*cc(β,δ) + δβγ*cc(α,δ)
     *                                 + δαδ*cc(β,γ) + δβδ*cc(α,γ) + δγδ*cc(α,β))
     *
     * And the f term: -15*m_j*(-7)*cccc(α,β,γ,δ)*f_j = 105*m_j*cccc(α,β,γ,δ)*f_j
     *
     * h term: 3*m_j*h_j*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)
     *------------------------------------------------------------------*/

    for (int k = 0; k <= K; ++k) {
        double* Ck = Cc + k * SZ3;

        for (int alpha = 0; alpha < 3; ++alpha) {
            int i = 3 + alpha;
            for (int beta = 0; beta < 3; ++beta) {
                for (int gamma = 0; gamma < 3; ++gamma) {
                    for (int delta = 0; delta < 3; ++delta) {
                        double val = 0.0;
                        for (int body = 0; body < 2; ++body) {
                            double mj = (body == 0) ? mu1 : mu;

                            /* h term */
                            double h_k = (body == 0) ? h1c[k] : h2c[k];
                            double hterm = dkr(alpha,beta)*dkr(gamma,delta)
                                         + dkr(alpha,gamma)*dkr(beta,delta)
                                         + dkr(beta,gamma)*dkr(alpha,delta);

                            /* e term: sum over all 6 pairings of (α,β,γ,δ) */
                            double eterm = dkr(alpha,beta)*get_cce(body,gamma,delta,k)
                                         + dkr(alpha,gamma)*get_cce(body,beta,delta,k)
                                         + dkr(beta,gamma)*get_cce(body,alpha,delta,k)
                                         + dkr(alpha,delta)*get_cce(body,beta,gamma,k)
                                         + dkr(beta,delta)*get_cce(body,alpha,gamma,k)
                                         + dkr(gamma,delta)*get_cce(body,alpha,beta,k);

                            /* f term */
                            double fterm = get_ccccf(body,alpha,beta,gamma,delta,k);

                            val += mj * (3.0*hterm*h_k - 15.0*eterm + 105.0*fterm);
                        }
                        Ck[idx4(i, beta, gamma, delta)] = val;
                    }
                }
            }
        }
    }

    /*------------------------------------------------------------------
     * FOURTH SPATIAL DERIVATIVE  D(i,p,q,r,s) = ∂⁴f_i / (∂X_p ∂X_q ∂X_r ∂X_s)
     *
     * Differentiating C w.r.t. c_epsilon:
     *
     *   D_j(3+α,β,γ,δ,ε) =
     *     3*m_j*(-5)*e_j * (δαβ*δγδ + δαγ*δβδ + δβγ*δαδ) * c_ε
     *   - 15*m_j*{
     *       [sum of 6 δ-cc terms differentiated w.r.t. c_ε]
     *     }
     *   + 105*m_j*{
     *       [d/d_c_ε of cccc*f]
     *     }
     *
     * Let me work through each piece:
     *
     * h term derivative:
     *   ∂/∂c_ε [h_j] = -5*c_ε*e_j
     *   So: 3*m_j*(-5)*c_ε*e_j * (δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)
     *
     * e term derivative (for a typical δ(a,b)*c_c*c_d*e term):
     *   ∂/∂c_ε [c_c*c_d*e] = δ(c,ε)*c_d*e + c_c*δ(d,ε)*e + c_c*c_d*(-7*c_ε*f)
     *                       = (δ(c,ε)*c_d + δ(d,ε)*c_c)*e - 7*c_c*c_d*c_ε*f
     *
     * f term derivative:
     *   ∂/∂c_ε [c_α*c_β*c_γ*c_δ*f]
     *     = (δαε*c_β*c_γ*c_δ + δβε*c_α*c_γ*c_δ + δγε*c_α*c_β*c_δ + δδε*c_α*c_β*c_γ)*f
     *       + c_α*c_β*c_γ*c_δ*(-9*c_ε*g9)
     *
     * But we don't have s^(-11/2). So we use the formula differently.
     * Actually for D we need s^(-9/2) which is f, and the derivative of f gives s^(-11/2).
     * However we can express D without s^(-11/2) by computing it directly from the
     * existing products.
     *
     * Let's define the full D formula using the pattern:
     *
     *   D_j(3+α,β,γ,δ,ε) =
     *     TERMS with e:
     *       -15*m_j*e_j * [sum of all 15 products of pairs of Kronecker deltas * remaining single coord]
     *     TERMS with f:
     *       +105*m_j*f_j * [sum of all 10 products of single delta * remaining triple coord product]
     *     TERMS with g9 (s^{-11/2}):
     *       -945*m_j*g9 * c_α*c_β*c_γ*c_δ*c_ε
     *
     * We need s^(-11/2). Let me add that.
     *
     * Actually, the general pattern for the n-th derivative of -m/r is well-known.
     * For the potential -m*r^{-1}, the n-th spatial derivative involves s^{-(2n+1)/2}.
     *
     * For D (4th derivative), we need up to s^{-11/2} = s^{-5.5}.
     *
     * Let me think about this differently. The D tensor for one body is:
     *
     * D_j(α,β,γ,δ,ε) = ∂⁴/∂c_β∂c_γ∂c_δ∂c_ε [ -m_j*c_α*g_j + correction ]
     *
     * The full pattern for derivatives of the gravitational potential term is:
     *
     * ∂^n / ∂c_{i1}...∂c_{in} [ c_α * s^{-3/2} ]
     *
     * This is a well-known recurrence. For each derivative, you differentiate
     * and get two terms: one from the coordinate factor and one from the power.
     *
     * Rather than derive the full 4th derivative formula, I'll compute D
     * numerically by differentiating C.
     *
     * Actually, let me just add the s^{-11/2} power and compute D explicitly.
     *------------------------------------------------------------------*/

    /* We need s^(-11/2) = s^{-5.5} for the D tensor. */
    TaylorScalar g9_1(K), g9_2(K);  /* s^{-11/2} for body 1 and 2 */

    /* Build them */
    g9_1[0] = std::pow(s1c[0], -5.5);
    g9_2[0] = std::pow(s2c[0], -5.5);
    for (int k = 1; k <= K; ++k) {
        auto power_coeff = [](const TaylorScalar& sc, TaylorScalar& gc,
                              double alpha, int kk) {
            double inv0 = 1.0 / sc[0];
            double s = 0.0;
            for (int j = 0; j < kk; ++j)
                s += (alpha * (kk - j) - j) * sc[kk - j] * gc[j];
            gc[kk] = inv0 * s / (double)kk;
        };
        power_coeff(s1c, g9_1, -5.5, k);
        power_coeff(s2c, g9_2, -5.5, k);
    }

    /* Five-coordinate products with g9 */
    /* We need c_α*c_β*c_γ*c_δ*c_ε * g9 */
    /* But building all 5-coord products with g9 is expensive.
     * Instead, let's use the formula directly. */

    /* For D, the formula for one body j is:
     *
     *  D_j(3+α,β,γ,δ,ε) =
     *    -15*m_j * [sum over all (15) ways to pick 3 deltas from {α,β,γ,δ,ε}
     *               times the remaining single coordinate * e_j]
     *  + 105*m_j * [sum over all (10) ways to pick 1 delta from pairs in {β,γ,δ,ε}
     *               combined with α, times the remaining cc product * f_j]
     *  - 945*m_j * c_α*c_β*c_γ*c_δ*c_ε * g9_j
     *
     * Actually this isn't quite right either. Let me use the systematic approach.
     *
     * The acceleration component for body j is:
     *   f_j(3+α) = -m_j * c_α * s_j^{-3/2}  (plus Coriolis for α=0,1)
     *
     * The purely gravitational part (ignoring Coriolis and centrifugal which are linear/constant):
     *   G_j(α) = -m_j * c_α * g_j      where g_j = s_j^{-3/2}
     *
     * The n-th partial derivative of G_j(α) w.r.t. position coordinates uses:
     *   ∂^n g_j / ∂c_{i1}...∂c_{in}
     *
     * The key recurrence for derivatives of s^p:
     *   ∂(s^p)/∂c_i = 2p * c_i * s^{p-1}
     *
     * For g = s^{-3/2}:
     *   ∂g/∂c_i = -3 * c_i * s^{-5/2} = -3 * c_i * h
     *   ∂²g/∂c_i∂c_j = -3*δ(i,j)*h + 15*c_i*c_j*e
     *   ∂³g/∂c_i∂c_j∂c_k = 15*(δ(i,j)*c_k + δ(i,k)*c_j + δ(j,k)*c_i)*e
     *                     - 105*c_i*c_j*c_k*f
     *   ∂⁴g/∂c_i∂c_j∂c_k∂c_l
     *     = 15*(δij*δkl + δik*δjl + δjk*δil)*e
     *     - 105*(δij*c_k*c_l + δik*c_j*c_l + δil*c_j*c_k
     *          + δjk*c_i*c_l + δjl*c_i*c_k + δkl*c_i*c_j)*f
     *     + 945*c_i*c_j*c_k*c_l * g9
     *
     * where g9 = s^{-11/2}.
     *
     * Now G_j(α) = -m_j * c_α * g_j, so:
     *   ∂G_j(α)/∂c_β = -m_j * (δαβ*g + c_α * ∂g/∂c_β)
     *                 = -m_j * (δαβ*g - 3*c_α*c_β*h)
     *                 [this gives A, matching what we have]
     *
     *   ∂²G_j(α)/∂c_β∂c_γ = -m_j * (δαβ*∂g/∂c_γ + δαγ*∂g/∂c_β + c_α*∂²g/∂c_β∂c_γ)
     *
     * Wait, that's not right. Let me be more careful:
     *   ∂/∂c_γ [δαβ*g + c_α*(-3*c_β*h)]
     *     = δαβ*∂g/∂c_γ + δαγ*(-3*c_β*h) + c_α*(-3)*(δβγ*h + c_β*∂h/∂c_γ)
     *     = δαβ*(-3*c_γ*h) - 3*δαγ*c_β*h - 3*δβγ*c_α*h + 15*c_α*c_β*c_γ*e
     *     = -3*(δαβ*c_γ + δαγ*c_β + δβγ*c_α)*h + 15*c_α*c_β*c_γ*e
     *
     *   B_j(3+α,β,γ) = -m_j * [-3*(δαβ*c_γ + δαγ*c_β + δβγ*c_α)*h + 15*c_α*c_β*c_γ*e]
     *                 = 3*m_j*(δαβ*c_γ + δαγ*c_β + δβγ*c_α)*h - 15*m_j*c_α*c_β*c_γ*e
     *   [matches!]
     *
     * For the 3rd derivative:
     *   ∂³G_j(α)/∂c_β∂c_γ∂c_δ = -m_j * ∂²/∂c_γ∂c_δ [δαβ*g + c_α*∂g/∂c_β]
     *
     * Using the chain rule on B (before the -m_j factor):
     *   ∂/∂c_δ [-3*(δαβ*c_γ + δαγ*c_β + δβγ*c_α)*h + 15*c_α*c_β*c_γ*e]
     *
     *   First term:
     *     -3*[δαβ*(δγδ*h + c_γ*∂h/∂c_δ) + δαγ*(δβδ*h + c_β*∂h/∂c_δ) + δβγ*(δαδ*h + c_α*∂h/∂c_δ)]
     *     = -3*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)*h
     *       -3*(δαβ*c_γ + δαγ*c_β + δβγ*c_α)*(-5*c_δ*e)
     *     = -3*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)*h
     *       +15*(δαβ*c_γ*c_δ + δαγ*c_β*c_δ + δβγ*c_α*c_δ)*e
     *
     *   Second term:
     *     15*(δαδ*c_β*c_γ + c_α*δβδ*c_γ + c_α*c_β*δγδ)*e
     *     + 15*c_α*c_β*c_γ*(-7*c_δ)*f
     *     = 15*(δαδ*c_β*c_γ + δβδ*c_α*c_γ + δγδ*c_α*c_β)*e
     *       - 105*c_α*c_β*c_γ*c_δ*f
     *
     *   Total (before -m_j):
     *     -3*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)*h
     *     +15*(δαβ*c_γ*c_δ + δαγ*c_β*c_δ + δβγ*c_α*c_δ + δαδ*c_β*c_γ + δβδ*c_α*c_γ + δγδ*c_α*c_β)*e
     *     -105*c_α*c_β*c_γ*c_δ*f
     *
     *   C_j(3+α,β,γ,δ) = -m_j * (above)
     *     = 3*m_j*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)*h
     *     - 15*m_j*(6 delta-cc terms)*e
     *     + 105*m_j*c_α*c_β*c_γ*c_δ*f
     *   [matches our earlier formula!]
     *
     * For the 4th derivative, differentiating C (before -m_j) w.r.t. c_ε:
     *
     *   h term: ∂/∂c_ε [-3*(3 delta-delta products)*h]
     *     = -3*(3 delta-delta products)*(-5*c_ε*e)
     *     = 15*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)*c_ε*e
     *
     *   e term: ∂/∂c_ε [15*(6 delta-cc terms)*e]
     *   For each δ(i,j)*c_k*c_l*e term:
     *     15*δ(i,j)*(δ(k,ε)*c_l + c_k*δ(l,ε))*e + 15*δ(i,j)*c_k*c_l*(-7*c_ε*f)
     *     = 15*δ(i,j)*(δkε*c_l + δlε*c_k)*e - 105*δ(i,j)*c_k*c_l*c_ε*f
     *
     *   f term: ∂/∂c_ε [-105*c_α*c_β*c_γ*c_δ*f]
     *     = -105*(δαε*c_β*c_γ*c_δ + δβε*c_α*c_γ*c_δ + δγε*c_α*c_β*c_δ + δδε*c_α*c_β*c_γ)*f
     *       -105*c_α*c_β*c_γ*c_δ*(-9*c_ε)*g9
     *     = -105*(4 terms)*f + 945*c_α*c_β*c_γ*c_δ*c_ε*g9
     *
     * Let me group by power:
     *
     * e terms: 15*(δαβ*δγδ + δαγ*δβδ + δβγ*δαδ)*c_ε*e
     *        + 15*[expanded delta-cc-delta terms]*e
     *
     * f terms: -105*[delta-cc-cc terms from e differentiation]*f
     *        - 105*(4 delta-ccc terms from f differentiation)*f
     *
     * g9 term: 945*c_α*c_β*c_γ*c_δ*c_ε*g9
     *
     * The full formula for D_j(3+α,β,γ,δ,ε) = -m_j * (above) is complex
     * but completely systematic. Let me implement it using the general
     * structure.
     *
     * D_j(3+α,β,γ,δ,ε) =
     *   -15*m_j * [SUM over the 15 pairings of deltas from {αβγδε}] * c_remaining * e
     *   +105*m_j * [SUM over the (4+6)=10 terms with single delta * cc * f]
     *   -945*m_j * ccccc * g9
     *
     * Actually, let me use the systematic approach:
     *
     * The n-th derivative of c_α * g (= c_α * s^{-3/2}) has the structure:
     *   (-1)^n * 1*3*5*...*(2n+1) * c_α*c_{i1}*...*c_{in} * s^{-(2n+3)/2}
     *   + lower order terms involving Kronecker deltas
     *
     * For the 4th derivative (β,γ,δ,ε), it's:
     *
     * D = -m_j * [
     *   δ_{αβ} * ∂³g/∂c_γ∂c_δ∂c_ε + δ_{αγ} * ∂³g/∂c_β∂c_δ∂c_ε
     * + δ_{αδ} * ∂³g/∂c_β∂c_γ∂c_ε + δ_{αε} * ∂³g/∂c_β∂c_γ∂c_δ
     * + c_α * ∂⁴g/∂c_β∂c_γ∂c_δ∂c_ε ]
     *
     * where we already know:
     *   ∂³g/∂c_i∂c_j∂c_k = 15*(δij*c_k+δik*c_j+δjk*c_i)*e - 105*c_i*c_j*c_k*f
     *
     *   ∂⁴g/∂c_β∂c_γ∂c_δ∂c_ε =
     *     15*(δβγ*δδε + δβδ*δγε + δβε*δγδ)*e
     *   - 105*(δβγ*c_δ*c_ε + δβδ*c_γ*c_ε + δβε*c_γ*c_δ
     *        + δγδ*c_β*c_ε + δγε*c_β*c_δ + δδε*c_β*c_γ)*f
     *   + 945*c_β*c_γ*c_δ*c_ε * g9
     *
     * This is the cleanest way. Let me implement it.
     *------------------------------------------------------------------*/

    /* We need c_α*c_β*c_γ*c_δ*c_ε * g9 products for D.
     * Rather than precomputing all 5-coord products, compute on the fly. */

    /* Helper for 5-coordinate product with g9 at order k */
    auto get_cccccg9 = [&](int body, int a, int b, int c, int d, int e_idx, int k) -> double {
        /* This is c_a * c_b * c_c * c_d * c_e * g9 at Taylor order k.
         * Compute via nested Cauchy products with precomputed 4-coord*f replaced by
         * rebuilding from scratch using nested convolutions. */
        /* We have 4-coord products with f. We need 5-coord with g9.
         * 5-coord*g9 = (c_a * cccc_bcde * ... ) - messy.
         * Instead, compute as convolution of c_a with (cccc*g9).
         * But we don't have cccc*g9 precomputed.
         * Let's compute as conv(c_a, conv(c_b, conv(c_c, conv(c_d, conv(c_e, g9))))).
         * This is O(k^5) per entry which is expensive but k is small (<=12).
         */

        /* For simplicity, use the precomputed 4-coord*f products and the relation:
         *   c_a*c_b*c_c*c_d*c_e * g9 can be computed as
         *   conv(coord_a, cccc_bcde_f * f^{-1} * g9) -- no, too complex.
         *
         * Better: just do the 5-level convolution directly. */
        const TaylorScalar& g9ref = (body == 0) ? g9_1 : g9_2;

        /* coord TaylorScalars */
        auto get_coord_ptr = [&](int body, int idx) -> const TaylorScalar& {
            if (body == 0) {
                if (idx == 0) return q1c;
                if (idx == 1) return yc;
                return zc;
            } else {
                if (idx == 0) return q2c;
                if (idx == 1) return yc;
                return zc;
            }
        };

        const TaylorScalar& ca = get_coord_ptr(body, a);
        const TaylorScalar& cb = get_coord_ptr(body, b);
        const TaylorScalar& cc_ts = get_coord_ptr(body, c);
        const TaylorScalar& cd = get_coord_ptr(body, d);
        const TaylorScalar& ce = get_coord_ptr(body, e_idx);

        /* 5-fold convolution at order k:
         * sum over j1+j2+j3+j4+j5+j6 = k of ca[j1]*cb[j2]*cc[j3]*cd[j4]*ce[j5]*g9[j6] */
        double result = 0.0;
        for (int j1 = 0; j1 <= k; ++j1)
            for (int j2 = 0; j2 <= k-j1; ++j2)
                for (int j3 = 0; j3 <= k-j1-j2; ++j3)
                    for (int j4 = 0; j4 <= k-j1-j2-j3; ++j4)
                        for (int j5 = 0; j5 <= k-j1-j2-j3-j4; ++j5) {
                            int j6 = k-j1-j2-j3-j4-j5;
                            result += ca[j1]*cb[j2]*cc_ts[j3]*cd[j4]*ce[j5]*g9ref[j6];
                        }
        return result;
    };

    /* Build D for each order k */
    for (int k = 0; k <= K; ++k) {
        double* Dk = Dc + k * SZ4;

        for (int alpha = 0; alpha < 3; ++alpha) {
            int i = 3 + alpha;
            for (int beta = 0; beta < 3; ++beta) {
                for (int gamma = 0; gamma < 3; ++gamma) {
                    for (int delta = 0; delta < 3; ++delta) {
                        for (int eps = 0; eps < 3; ++eps) {
                            /* D_j(3+α,β,γ,δ,ε) = -m_j * [
                             *   δαβ * ∂³g/∂c_γ∂c_δ∂c_ε + δαγ * ∂³g/∂c_β∂c_δ∂c_ε
                             * + δαδ * ∂³g/∂c_β∂c_γ∂c_ε + δαε * ∂³g/∂c_β∂c_γ∂c_δ
                             * + c_α * ∂⁴g/∂c_β∂c_γ∂c_δ∂c_ε ]
                             */

                            /* ∂³g/∂c_i∂c_j∂c_k at order k:
                             *   15*(δij*c_k+δik*c_j+δjk*c_i)*e - 105*c_i*c_j*c_k*f
                             */
                            auto d3g = [&](int body, int ii, int jj, int kk, int ord) -> double {
                                double eterm = dkr(ii,jj)*get_ce(body,kk,ord)
                                             + dkr(ii,kk)*get_ce(body,jj,ord)
                                             + dkr(jj,kk)*get_ce(body,ii,ord);
                                double fterm = get_cccf(body,ii,jj,kk,ord);
                                return 15.0*eterm - 105.0*fterm;
                            };

                            double val = 0.0;
                            for (int body = 0; body < 2; ++body) {
                                double mj = (body == 0) ? mu1 : mu;

                                /* The 4 terms with δ_{α,x} * ∂³g */
                                double sum_d3g = 0.0;
                                sum_d3g += dkr(alpha,beta)  * d3g(body, gamma, delta, eps, k);
                                sum_d3g += dkr(alpha,gamma) * d3g(body, beta, delta, eps, k);
                                sum_d3g += dkr(alpha,delta) * d3g(body, beta, gamma, eps, k);
                                sum_d3g += dkr(alpha,eps)   * d3g(body, beta, gamma, delta, k);

                                /* c_α * ∂⁴g term: expand the product c_α * G4
                                 * using precomputed multi-coordinate Taylor products:
                                 *   (c_α * G4)[k] = 15*(3 δδ terms)*(c_α*e)[k]
                                 *                 - 105*(6 δ terms)*(c_α*cc*f)[k]
                                 *                 + 945*(c_α*c_β*c_γ*c_δ*c_ε*g9)[k]
                                 */
                                double dterm4 = (dkr(beta,gamma)*dkr(delta,eps)
                                               + dkr(beta,delta)*dkr(gamma,eps)
                                               + dkr(beta,eps)*dkr(gamma,delta));

                                double ca_e_k = get_ce(body, alpha, k);

                                double caG4_k = 15.0 * dterm4 * ca_e_k
                                    - 105.0 * (
                                        dkr(beta,gamma)*get_cccf(body,alpha,delta,eps,k)
                                      + dkr(beta,delta)*get_cccf(body,alpha,gamma,eps,k)
                                      + dkr(beta,eps)*get_cccf(body,alpha,gamma,delta,k)
                                      + dkr(gamma,delta)*get_cccf(body,alpha,beta,eps,k)
                                      + dkr(gamma,eps)*get_cccf(body,alpha,beta,delta,k)
                                      + dkr(delta,eps)*get_cccf(body,alpha,beta,gamma,k)
                                    )
                                    + 945.0 * get_cccccg9(body,alpha,beta,gamma,delta,eps,k);

                                val += -mj * (sum_d3g + caG4_k);
                            }
                            Dk[idx5(i, beta, gamma, delta, eps)] = val;
                        }
                    }
                }
            }
        }
    }

    /*==================================================================
     * STEP 3: Propagate Taylor coefficients of Phi1, Phi2, Phi3, Phi4
     * using the variational equations in Taylor-coefficient form.
     *
     * For a variational equation of the form
     *   dPhi/dt = RHS(t)
     * the Taylor coefficient recurrence is:
     *   Phi_c[k+1] = RHS_c[k] / (k+1)
     *
     * where RHS_c[k] involves Cauchy products of A/B/C/D Taylor
     * coefficient arrays with Phi Taylor coefficient arrays.
     *==================================================================*/

    /* Allocate Taylor coefficient arrays for Phi1..Phi4.
     * We need K+1 slices (orders 0..K). */
    std::vector<double> P1c(SZ1 * Kp1, 0.0);
    std::vector<double> P2c(SZ2 * Kp1, 0.0);
    std::vector<double> P3c(SZ3 * Kp1, 0.0);
    std::vector<double> P4c(SZ4 * Kp1, 0.0);

    /* Set initial conditions (k=0 slice) */
    std::memcpy(P1c.data(), Phi1_0, SZ1 * sizeof(double));
    std::memcpy(P2c.data(), Phi2_0, SZ2 * sizeof(double));
    std::memcpy(P3c.data(), Phi3_0, SZ3 * sizeof(double));
    std::memcpy(P4c.data(), Phi4_0, SZ4 * sizeof(double));

    /* Iterate: for each k = 0..K-1, compute RHS at order k, then set Phi[k+1]. */
    for (int k = 0; k < K; ++k) {
        double inv_kp1 = 1.0 / (double)(k + 1);

        /*--------------------------------------------------------------
         * Phi1 variational equation:
         *   d/dt Phi1(i,a) = A(i,p) * Phi1(p,a)
         *
         * Taylor coefficient form:
         *   Phi1c[k+1](i,a) = (1/(k+1)) * sum_{j=0}^{k} A[j](i,p) * Phi1c[k-j](p,a)
         *--------------------------------------------------------------*/
        for (int i = 0; i < 6; ++i) {
            for (int a = 0; a < 6; ++a) {
                double rhs = 0.0;
                for (int j = 0; j <= k; ++j) {
                    const double* Aj = Ac + j * SZ1;
                    const double* P1j = P1c.data() + (k-j) * SZ1;
                    for (int p = 0; p < 6; ++p) {
                        rhs += Aj[idx2(i,p)] * P1j[idx2(p,a)];
                    }
                }
                P1c[(k+1)*SZ1 + idx2(i,a)] = rhs * inv_kp1;
            }
        }

        /*--------------------------------------------------------------
         * Phi2 variational equation:
         *   d/dt Phi2(i,a,b) = A(i,p)*Phi2(p,a,b)
         *                    + B(i,p,q)*Phi1(p,a)*Phi1(q,b)
         *--------------------------------------------------------------*/
        for (int i = 0; i < 6; ++i) {
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                    double rhs = 0.0;

                    /* A*Phi2 term */
                    for (int j = 0; j <= k; ++j) {
                        const double* Aj = Ac + j * SZ1;
                        const double* P2j = P2c.data() + (k-j) * SZ2;
                        for (int p = 0; p < 6; ++p)
                            rhs += Aj[idx2(i,p)] * P2j[idx3(p,a,b)];
                    }

                    /* B*Phi1*Phi1 term: triple Cauchy product */
                    /* sum_{j1+j2+j3=k} B[j1](i,p,q) * P1[j2](p,a) * P1[j3](q,b) */
                    for (int p = 0; p < 6; ++p) {
                        for (int q = 0; q < 6; ++q) {
                            for (int j1 = 0; j1 <= k; ++j1) {
                                double Bval = Bc[j1*SZ2 + idx3(i,p,q)];
                                if (Bval == 0.0) continue;
                                for (int j2 = 0; j2 <= k-j1; ++j2) {
                                    int j3 = k - j1 - j2;
                                    rhs += Bval
                                         * P1c[j2*SZ1 + idx2(p,a)]
                                         * P1c[j3*SZ1 + idx2(q,b)];
                                }
                            }
                        }
                    }

                    P2c[(k+1)*SZ2 + idx3(i,a,b)] = rhs * inv_kp1;
                }
            }
        }

        /*--------------------------------------------------------------
         * Phi3 variational equation:
         *   d/dt Phi3(i,a,b,c) = A(i,p)*Phi3(p,a,b,c)
         *     + B(i,p,q)*[Phi2(p,a,b)*Phi1(q,c)
         *               + Phi2(p,a,c)*Phi1(q,b)
         *               + Phi2(p,b,c)*Phi1(q,a)]
         *     + C(i,p,q,r)*Phi1(p,a)*Phi1(q,b)*Phi1(r,c)
         *--------------------------------------------------------------*/
        for (int i = 0; i < 6; ++i) {
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                    for (int c = 0; c < 6; ++c) {
                        double rhs = 0.0;

                        /* A*Phi3 */
                        for (int j = 0; j <= k; ++j) {
                            const double* Aj = Ac + j * SZ1;
                            const double* P3j = P3c.data() + (k-j) * SZ3;
                            for (int p = 0; p < 6; ++p)
                                rhs += Aj[idx2(i,p)] * P3j[idx4(p,a,b,c)];
                        }

                        /* B*(Phi2*Phi1) terms: 3 permutations */
                        for (int p = 0; p < 6; ++p) {
                            for (int q = 0; q < 6; ++q) {
                                for (int j1 = 0; j1 <= k; ++j1) {
                                    double Bval = Bc[j1*SZ2 + idx3(i,p,q)];
                                    if (Bval == 0.0) continue;
                                    for (int j2 = 0; j2 <= k-j1; ++j2) {
                                        int j3 = k - j1 - j2;
                                        rhs += Bval * (
                                            P2c[j2*SZ2+idx3(p,a,b)] * P1c[j3*SZ1+idx2(q,c)]
                                          + P2c[j2*SZ2+idx3(p,a,c)] * P1c[j3*SZ1+idx2(q,b)]
                                          + P2c[j2*SZ2+idx3(p,b,c)] * P1c[j3*SZ1+idx2(q,a)]
                                        );
                                    }
                                }
                            }
                        }

                        /* C*Phi1*Phi1*Phi1: 4-fold Cauchy product */
                        for (int p = 0; p < 6; ++p) {
                            for (int q = 0; q < 6; ++q) {
                                for (int r = 0; r < 6; ++r) {
                                    for (int j1 = 0; j1 <= k; ++j1) {
                                        double Cval = Cc[j1*SZ3 + idx4(i,p,q,r)];
                                        if (Cval == 0.0) continue;
                                        for (int j2 = 0; j2 <= k-j1; ++j2) {
                                            for (int j3 = 0; j3 <= k-j1-j2; ++j3) {
                                                int j4 = k-j1-j2-j3;
                                                rhs += Cval
                                                     * P1c[j2*SZ1+idx2(p,a)]
                                                     * P1c[j3*SZ1+idx2(q,b)]
                                                     * P1c[j4*SZ1+idx2(r,c)];
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        P3c[(k+1)*SZ3 + idx4(i,a,b,c)] = rhs * inv_kp1;
                    }
                }
            }
        }

        /*--------------------------------------------------------------
         * Phi4 variational equation:
         *   d/dt Phi4(i,a,b,c,d) = A(i,p)*Phi4(p,a,b,c,d)
         *     + B(i,p,q)*[Phi3(p,a,b,c)*Phi1(q,d)
         *               + Phi3(p,a,b,d)*Phi1(q,c)
         *               + Phi3(p,a,c,d)*Phi1(q,b)
         *               + Phi3(p,b,c,d)*Phi1(q,a)
         *               + Phi2(p,a,b)*Phi2(q,c,d)
         *               + Phi2(p,a,c)*Phi2(q,b,d)
         *               + Phi2(p,a,d)*Phi2(q,b,c)]
         *     + C(i,p,q,r)*[Phi2(p,a,b)*Phi1(q,c)*Phi1(r,d)
         *                 + Phi2(p,a,c)*Phi1(q,b)*Phi1(r,d)
         *                 + Phi2(p,a,d)*Phi1(q,b)*Phi1(r,c)
         *                 + Phi2(p,b,c)*Phi1(q,a)*Phi1(r,d)
         *                 + Phi2(p,b,d)*Phi1(q,a)*Phi1(r,c)
         *                 + Phi2(p,c,d)*Phi1(q,a)*Phi1(r,b)]
         *     + D(i,p,q,r,s)*Phi1(p,a)*Phi1(q,b)*Phi1(r,c)*Phi1(s,d)
         *--------------------------------------------------------------*/
        for (int i = 0; i < 6; ++i) {
            for (int a = 0; a < 6; ++a) {
                for (int b = 0; b < 6; ++b) {
                    for (int c = 0; c < 6; ++c) {
                        for (int d = 0; d < 6; ++d) {
                            double rhs = 0.0;

                            /* A*Phi4 */
                            for (int j = 0; j <= k; ++j) {
                                const double* Aj = Ac + j * SZ1;
                                const double* P4j = P4c.data() + (k-j) * SZ4;
                                for (int p = 0; p < 6; ++p)
                                    rhs += Aj[idx2(i,p)] * P4j[idx5(p,a,b,c,d)];
                            }

                            /* B terms */
                            for (int p = 0; p < 6; ++p) {
                                for (int q = 0; q < 6; ++q) {
                                    for (int j1 = 0; j1 <= k; ++j1) {
                                        double Bval = Bc[j1*SZ2 + idx3(i,p,q)];
                                        if (Bval == 0.0) continue;
                                        for (int j2 = 0; j2 <= k-j1; ++j2) {
                                            int j3 = k - j1 - j2;
                                            /* Phi3*Phi1 terms */
                                            rhs += Bval * (
                                                P3c[j2*SZ3+idx4(p,a,b,c)] * P1c[j3*SZ1+idx2(q,d)]
                                              + P3c[j2*SZ3+idx4(p,a,b,d)] * P1c[j3*SZ1+idx2(q,c)]
                                              + P3c[j2*SZ3+idx4(p,a,c,d)] * P1c[j3*SZ1+idx2(q,b)]
                                              + P3c[j2*SZ3+idx4(p,b,c,d)] * P1c[j3*SZ1+idx2(q,a)]
                                            );
                                            /* Phi2*Phi2 terms */
                                            rhs += Bval * (
                                                P2c[j2*SZ2+idx3(p,a,b)] * P2c[j3*SZ2+idx3(q,c,d)]
                                              + P2c[j2*SZ2+idx3(p,a,c)] * P2c[j3*SZ2+idx3(q,b,d)]
                                              + P2c[j2*SZ2+idx3(p,a,d)] * P2c[j3*SZ2+idx3(q,b,c)]
                                            );
                                        }
                                    }
                                }
                            }

                            /* C terms */
                            for (int p = 0; p < 6; ++p) {
                                for (int q = 0; q < 6; ++q) {
                                    for (int r = 0; r < 6; ++r) {
                                        for (int j1 = 0; j1 <= k; ++j1) {
                                            double Cval = Cc[j1*SZ3 + idx4(i,p,q,r)];
                                            if (Cval == 0.0) continue;
                                            for (int j2 = 0; j2 <= k-j1; ++j2) {
                                                for (int j3 = 0; j3 <= k-j1-j2; ++j3) {
                                                    int j4 = k-j1-j2-j3;
                                                    rhs += Cval * (
                                                        P2c[j2*SZ2+idx3(p,a,b)] * P1c[j3*SZ1+idx2(q,c)] * P1c[j4*SZ1+idx2(r,d)]
                                                      + P2c[j2*SZ2+idx3(p,a,c)] * P1c[j3*SZ1+idx2(q,b)] * P1c[j4*SZ1+idx2(r,d)]
                                                      + P2c[j2*SZ2+idx3(p,a,d)] * P1c[j3*SZ1+idx2(q,b)] * P1c[j4*SZ1+idx2(r,c)]
                                                      + P2c[j2*SZ2+idx3(p,b,c)] * P1c[j3*SZ1+idx2(q,a)] * P1c[j4*SZ1+idx2(r,d)]
                                                      + P2c[j2*SZ2+idx3(p,b,d)] * P1c[j3*SZ1+idx2(q,a)] * P1c[j4*SZ1+idx2(r,c)]
                                                      + P2c[j2*SZ2+idx3(p,c,d)] * P1c[j3*SZ1+idx2(q,a)] * P1c[j4*SZ1+idx2(r,b)]
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            /* D terms */
                            for (int p = 0; p < 6; ++p) {
                                for (int q = 0; q < 6; ++q) {
                                    for (int r = 0; r < 6; ++r) {
                                        for (int s = 0; s < 6; ++s) {
                                            for (int j1 = 0; j1 <= k; ++j1) {
                                                double Dval = Dc[j1*SZ4 + idx5(i,p,q,r,s)];
                                                if (Dval == 0.0) continue;
                                                for (int j2 = 0; j2 <= k-j1; ++j2) {
                                                    for (int j3 = 0; j3 <= k-j1-j2; ++j3) {
                                                        for (int j4 = 0; j4 <= k-j1-j2-j3; ++j4) {
                                                            int j5 = k-j1-j2-j3-j4;
                                                            rhs += Dval
                                                                 * P1c[j2*SZ1+idx2(p,a)]
                                                                 * P1c[j3*SZ1+idx2(q,b)]
                                                                 * P1c[j4*SZ1+idx2(r,c)]
                                                                 * P1c[j5*SZ1+idx2(s,d)];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            P4c[(k+1)*SZ4 + idx5(i,a,b,c,d)] = rhs * inv_kp1;
                        }
                    }
                }
            }
        }
    } /* end of k loop */

    /*==================================================================
     * STEP 4: Convert Taylor coefficients to raw time derivatives.
     *
     *   X^(k) = k! * Xc[k]
     *   Phi^(k) = k! * Phic[k]
     *
     * Output layout (column-major, MATLAB style):
     *   Xders(i,k) = Xders[i + 6*k]
     *   Phi1ders(i,a,k) = Phi1ders[i + 6*(a + 6*k)]
     *   etc.
     *==================================================================*/

    for (int k = 0; k <= K; ++k) {
        double fk = fact_table[k];

        /* State derivatives */
        Xders[0 + 6*k] = fk * xc[k];
        Xders[1 + 6*k] = fk * yc[k];
        Xders[2 + 6*k] = fk * zc[k];
        Xders[3 + 6*k] = fk * vxc[k];
        Xders[4 + 6*k] = fk * vyc[k];
        Xders[5 + 6*k] = fk * vzc[k];

        /* Phi1 derivatives */
        for (int idx = 0; idx < SZ1; ++idx)
            Phi1ders[idx + SZ1*k] = fk * P1c[k*SZ1 + idx];

        /* Phi2 derivatives */
        for (int idx = 0; idx < SZ2; ++idx)
            Phi2ders[idx + SZ2*k] = fk * P2c[k*SZ2 + idx];

        /* Phi3 derivatives */
        for (int idx = 0; idx < SZ3; ++idx)
            Phi3ders[idx + SZ3*k] = fk * P3c[k*SZ3 + idx];

        /* Phi4 derivatives */
        for (int idx = 0; idx < SZ4; ++idx)
            Phi4ders[idx + SZ4*k] = fk * P4c[k*SZ4 + idx];
    }

    return 0;  /* success */
}
