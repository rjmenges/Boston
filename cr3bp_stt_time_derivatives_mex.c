/*==========================================================================
 * cr3bp_stt_time_derivatives_mex.c
 *
 * MATLAB MEX gateway (pure C) for computing higher-order time derivatives
 * of the CR3BP state, STM, and 2nd-4th order State Transition Tensors.
 *
 * This file is intentionally a .c file (not .cpp) so that MATLAB's mex
 * compiler uses the legacy C MEX API with the mexFunction entry point.
 * The heavy computation lives in cr3bp_stt_time_derivatives.cpp (C++),
 * linked via extern "C".
 *
 * USAGE:
 *   [Xders, Phi1ders, Phi2ders, Phi3ders, Phi4ders] = ...
 *       cr3bp_stt_time_derivatives_mex(X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0, mu, N)
 *
 * INPUTS:
 *   X0       6 x 1                        state at t0
 *   Phi1_0   6 x 6                        STM at t0
 *   Phi2_0   6 x 6 x 6                    2nd STT at t0
 *   Phi3_0   6 x 6 x 6 x 6                3rd STT at t0
 *   Phi4_0   6 x 6 x 6 x 6 x 6            4th STT at t0
 *   mu       scalar                        CR3BP mass parameter
 *   N        scalar integer                max derivative order (0..12)
 *
 * OUTPUTS:
 *   Xders      6 x (N+1)                  raw time derivatives X^(k)
 *   Phi1ders   6 x 6 x (N+1)              raw Phi1^(k)
 *   Phi2ders   6 x 6 x 6 x (N+1)          raw Phi2^(k)
 *   Phi3ders   6 x 6 x 6 x 6 x (N+1)      raw Phi3^(k)
 *   Phi4ders   6 x 6 x 6 x 6 x 6 x (N+1)  raw Phi4^(k)
 *
 * COMPILE:
 *   mex -O cr3bp_stt_time_derivatives_mex.c cr3bp_stt_time_derivatives.cpp
 *
 *=========================================================================*/

#include "mex.h"
#include "cr3bp_stt_time_derivatives.h"

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    int N, Np1, rc;
    double mu;
    const double *X0, *Phi1_0, *Phi2_0, *Phi3_0, *Phi4_0;
    double *Xders, *Phi1ders, *Phi2ders, *Phi3ders, *Phi4ders;

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
    X0 = mxGetPr(prhs[0]);

    /* Phi1_0: 6 x 6 = 36 elements */
    if (mxGetNumberOfElements(prhs[1]) != 36)
        mexErrMsgIdAndTxt("CR3BP:Phi1", "Phi1_0 must have 36 elements (6x6)");
    Phi1_0 = mxGetPr(prhs[1]);

    /* Phi2_0: 6 x 6 x 6 = 216 elements */
    if (mxGetNumberOfElements(prhs[2]) != 216)
        mexErrMsgIdAndTxt("CR3BP:Phi2", "Phi2_0 must have 216 elements (6x6x6)");
    Phi2_0 = mxGetPr(prhs[2]);

    /* Phi3_0: 6 x 6 x 6 x 6 = 1296 elements */
    if (mxGetNumberOfElements(prhs[3]) != 1296)
        mexErrMsgIdAndTxt("CR3BP:Phi3", "Phi3_0 must have 1296 elements (6x6x6x6)");
    Phi3_0 = mxGetPr(prhs[3]);

    /* Phi4_0: 6 x 6 x 6 x 6 x 6 = 7776 elements */
    if (mxGetNumberOfElements(prhs[4]) != 7776)
        mexErrMsgIdAndTxt("CR3BP:Phi4", "Phi4_0 must have 7776 elements (6x6x6x6x6)");
    Phi4_0 = mxGetPr(prhs[4]);

    /* mu: scalar */
    if (!mxIsScalar(prhs[5]))
        mexErrMsgIdAndTxt("CR3BP:mu", "mu must be a scalar");
    mu = mxGetScalar(prhs[5]);

    /* N: scalar integer */
    if (!mxIsScalar(prhs[6]))
        mexErrMsgIdAndTxt("CR3BP:N", "N must be a scalar");
    N = (int)mxGetScalar(prhs[6]);
    if (N < 0 || N > CR3BP_NMAX)
        mexErrMsgIdAndTxt("CR3BP:N", "N must be in [0, %d]", CR3BP_NMAX);

    Np1 = N + 1;

    /* ---- Allocate outputs ---- */
    {
        mwSize Xdims[2]  = {6, (mwSize)Np1};
        mwSize P1dims[3] = {6, 6, (mwSize)Np1};
        mwSize P2dims[4] = {6, 6, 6, (mwSize)Np1};
        mwSize P3dims[5] = {6, 6, 6, 6, (mwSize)Np1};
        mwSize P4dims[6] = {6, 6, 6, 6, 6, (mwSize)Np1};

        plhs[0] = mxCreateNumericArray(2, Xdims,  mxDOUBLE_CLASS, mxREAL);
        plhs[1] = mxCreateNumericArray(3, P1dims, mxDOUBLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericArray(4, P2dims, mxDOUBLE_CLASS, mxREAL);
        plhs[3] = mxCreateNumericArray(5, P3dims, mxDOUBLE_CLASS, mxREAL);
        plhs[4] = mxCreateNumericArray(6, P4dims, mxDOUBLE_CLASS, mxREAL);
    }

    Xders    = mxGetPr(plhs[0]);
    Phi1ders = mxGetPr(plhs[1]);
    Phi2ders = mxGetPr(plhs[2]);
    Phi3ders = mxGetPr(plhs[3]);
    Phi4ders = mxGetPr(plhs[4]);

    /* ---- Call the computation ---- */
    rc = compute_cr3bp_stt_taylor_coefficients(
            X0, Phi1_0, Phi2_0, Phi3_0, Phi4_0,
            mu, N,
            Xders, Phi1ders, Phi2ders, Phi3ders, Phi4ders);

    if (rc != 0)
        mexErrMsgIdAndTxt("CR3BP:compute", "Computation failed (rc=%d)", rc);
}
