/* spouterprod.c
 * Bert Huang
 * bert@cs.columbia.edu
 *
 * This mex function provides a fast sparse outer product, which is 
 * useful when you need sparse entries in a large, dense outer product.
 * The input is a sparse mask, and full matrices U and V
 * The output is a sparse matrix with nonzeros where the mask is nonzero, 
 * and is equivelent to the matlab expression mask.*(U*V')
 *
 * Unfortunately matlab does not compute the above expression efficiently, 
 * so this mex file is helpful.
 *
 * Compile by typing 'mex spouterprod.c'
 *
 * Copyright 2009, 2010 Bert Huang
 * Feel free to contact the author with any questions or suggestions.
 *
 *
 * Updates:
 *
 * 5/27/10 - updated to support 64-bit matlab. Compile with -largeArrayDims
 *
 * Known Issues:
 *
 * - Code does not take advantage of multithreading
 * - Does not support sparse U and V
 *
 * This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */



#include "mex.h"


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize *ir, *jc, *ir_out, *jc_out, *uir, *ujc, *vir, *vjc;
    int nnz=0, M, N, D, i, j, 
            row, column;
    double *U, *V, *out, *data;
    
    if (nrhs != 3) 
        mexErrMsgTxt("Input error: Expected sparse mask, U, V, 1 output");
    if (!mxIsSparse(prhs[0]))
        mexErrMsgTxt("Error: Mask must be sparse\n");
    if (!mxIsSparse(prhs[1]) || !mxIsSparse(prhs[2]))
        mexErrMsgTxt("Error: sparse U or V required");
    
    /* load input */
    ir = mxGetIr(prhs[0]);
    jc = mxGetJc(prhs[0]);
    M = mxGetM(prhs[0]);
    N = mxGetN(prhs[0]);
    nnz = jc[N];
    
    if (mxGetN(prhs[1])==M && mxGetN(prhs[2])==N) {
        D = mxGetM(prhs[1]);
        U = mxGetPr(prhs[1]);
        V = mxGetPr(prhs[2]);
    } else 
        mexErrMsgTxt("Error: Matrix sizes are incorrect");
    
    uir = mxGetIr(prhs[1]);
    vir = mxGetIr(prhs[2]);
    ujc = mxGetJc(prhs[1]);
    vjc = mxGetJc(prhs[2]);
    
    /* open output  */
    
    plhs[0] = mxCreateSparse(M, N, nnz, 0);
    out = mxGetPr(plhs[0]);
    
    ir_out = mxGetIr(plhs[0]);
    jc_out = mxGetJc(plhs[0]);
    
    column=0;
    jc_out[0] = jc[0]; 
    for (i=0; i<nnz; i++) {
        row = ir[i];
        ir_out[i] = ir[i];        
        
        while (i>=jc[column+1]) {
            column++;
        }
        
        /* printf("Nonzero at %d, %d\n", row,column); */
        
        out[i] = 0;
        
        int k = vjc[column];
        
        /* for each nonzero element in U(row,:) */
        for (j = ujc[row]; j < ujc[row+1] && k < vjc[column+1]; j++) {
            while (k < vjc[column+1] && vir[k] < uir[j])
                k++;
            if (uir[j] == vir[k])
                out[i] += U[j]*V[k];
        }
        if (out[i] > 1e128)
            out[i] = 1e128;
    }
    
    for (i=0; i<N; i++) {
        jc_out[i] = jc[i];
    }
    jc_out[N] = nnz;

}


