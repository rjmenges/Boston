/*==========================================================================
 * cr3bp_stt_time_derivatives_mex.cpp
 *
 * MATLAB MEX interface for computing higher-order time derivatives of the
 * CR3BP state, STM, and 2nd-4th order State Transition Tensors.
 *
 * USAGE:
 *   [Xders, Phi1ders, Phi2ders, Phi3ders, Phi4ders] = ...
 *       cr3bp_stt_time_derivatives_mex(X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, N)
 *
 * INPUTS:
 *   X0       6 x 1           state at t0
 *   Phi1_0   6 x 6           STM at t0
 *   Phi2_0   6 x 6 x 6       2nd STT at t0
 *   Phi3_0   6 x 6 x 6 x 6   3rd STT at t0
 *   Phi4_0   6 x 6 x 6 x 6 x 6  4th STT at t0
 *   mu       scalar          CR3BP mass parameter
 *   N        scalar integer  max derivative order (0..12)
 *
 * OUTPUTS:
 *   Xders      6 x (N+1)                      raw time derivatives X^(k)
 *   Phi1ders   6 x 6 x (N+1)                  raw Phi1^(k)
 *   Phi2ders   6 x 6 x 6 x (N+1)              raw Phi2^(k)
 *   Phi3ders   6 x 6 x 6 x 6 x (N+1)          raw Phi3^(k)
 *   Phi4ders   6 x 6 x 6 x 6 x 6 x (N+1)      raw Phi4^(k)
 *
 * COMPILE (use -R2017b for the legacy C MEX API / mexFunction entry point):
 *   mex -O -R2017b cr3bp_stt_time_derivatives_mex.cpp cr3bp_stt_time_derivatives.cpp
 *
 *=========================================================================*/

#include "mex.h"
#include "cr3bp_stt_time_derivatives.h"
#include <cstring>

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* ---- Input validation ---- */
    if (nrhs != 7)
        mexErrMsgIdAndTxt("CR3BP:nrhs",
            "Seven inputs required: X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, N");
    if (nlhs > 5)
        mexErrMsgIdAndTxt("CR3BP:nlhs",
            "At most five outputs: Xders, Phi1ders, Phi2ders, Phi3ders, Phi4ders");

    /* ---- Parse inputs ---- */

    /* X0: 6 x 1 */
    if (mxGetNumberOfElements(prhs[0]) != 6)
        mexErrMsgIdAndTxt("CR3BP:X0", "X0 must have 6 elements");
    const double* X0 = mxGetPr(prhs[0]);

    /* Phi1_0: 6 x 6 = 36 elements */
    if (mxGetNumberOfElements(prhs[1]) != 36)
        mexErrMsgIdAndTxt("CR3BP:Phi1", "Phi1_0 must have 36 elements (6x6)");
    const double* Phi1_0 = mxGetPr(prhs[1]);

    /* Phi2_0: 6 x 6 x 6 = 216 elements */
    if (mxGetNumberOfElements(prhs[2]) != 216)
        mexErrMsgIdAndTxt("CR3BP:Phi2", "Phi2_0 must have 216 elements (6x6x6)");
    const double* Phi2_0 = mxGetPr(prhs[2]);

    /* Phi3_0: 6 x 6 x 6 x 6 = 1296 elements */
    if (mxGetNumberOfElements(prhs[3]) != 1296)
        mexErrMsgIdAndTxt("CR3BP:Phi3", "Phi3_0 must have 1296 elements (6x6x6x6)");
    const double* Phi3_0 = mxGetPr(prhs[3]);

    /* Phi4_0: 6 x 6 x 6 x 6 x 6 = 7776 elements */
    if (mxGetNumberOfElements(prhs[4]) != 7776)
        mexErrMsgIdAndTxt("CR3BP:Phi4", "Phi4_0 must have 7776 elements (6x6x6x6x6)");
    const double* Phi4_0 = mxGetPr(prhs[4]);

    /* mu: scalar */
    if (!mxIsScalar(prhs[5]))
        mexErrMsgIdAndTxt("CR3BP:mu", "mu must be a scalar");
    double mu = mxGetScalar(prhs[5]);

    /* N: scalar integer */
    if (!mxIsScalar(prhs[6]))
        mexErrMsgIdAndTxt("CR3BP:N", "N must be a scalar");
    int N = (int)mxGetScalar(prhs[6]);
    if (N < 0 || N > NMAX)
        mexErrMsgIdAndTxt("CR3BP:N", "N must be in [0, %d]", NMAX);

    int Np1 = N + 1;

    /* ---- Allocate outputs ---- */
    /* Xders: 6 x (N+1) */
    mwSize Xdims[2] = {6, (mwSize)Np1};
    plhs[0] = mxCreateNumericArray(2, Xdims, mxDOUBLE_CLASS, mxREAL);
    double* Xders = mxGetPr(plhs[0]);

    /* Phi1ders: 6 x 6 x (N+1) */
    mwSize P1dims[3] = {6, 6, (mwSize)Np1};
    plhs[1] = mxCreateNumericArray(3, P1dims, mxDOUBLE_CLASS, mxREAL);
    double* Phi1ders = mxGetPr(plhs[1]);

    /* Phi2ders: 6 x 6 x 6 x (N+1) */
    mwSize P2dims[4] = {6, 6, 6, (mwSize)Np1};
    plhs[2] = mxCreateNumericArray(4, P2dims, mxDOUBLE_CLASS, mxREAL);
    double* Phi2ders = mxGetPr(plhs[2]);

    /* Phi3ders: 6 x 6 x 6 x 6 x (N+1) */
    mwSize P3dims[5] = {6, 6, 6, 6, (mwSize)Np1};
    plhs[3] = mxCreateNumericArray(5, P3dims, mxDOUBLE_CLASS, mxREAL);
    double* Phi3ders = mxGetPr(plhs[3]);

    /* Phi4ders: 6 x 6 x 6 x 6 x 6 x (N+1) */
    mwSize P4dims[6] = {6, 6, 6, 6, 6, (mwSize)Np1};
    plhs[4] = mxCreateNumericArray(6, P4dims, mxDOUBLE_CLASS, mxREAL);
    double* Phi4ders = mxGetPr(plhs[4]);

    /* ---- Call the computation ---- */
    try {
        compute_cr3bp_stt_taylor_coefficients(
            X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0,
            mu, N,
            Xders, Phi1ders, Phi2ders, Phi3ders, Phi4ders);
    }
    catch (const std::exception& e) {
        mexErrMsgIdAndTxt("CR3BP:compute", "%s", e.what());
    }
}
