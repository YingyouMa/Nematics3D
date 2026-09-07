#ifndef QDIAG_KERNEL_H
#define QDIAG_KERNEL_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Diagonalize the traceless symmetric matrix
 *
 *   Q = [ qxx  qxy  qxz ]
 *       [ qxy  qyy  qyz ]
 *       [ qxz  qyz -qxx-qyy ]
 *
 * Preconditions for the fast physical-Q kernel:
 *   - finite IEEE-754 double inputs;
 *   - a valid symmetric traceless Q tensor;
 *   - physical components are expected to lie in [-1, 1].
 *
 * Outputs:
 *   eval[0] <= eval[1] <= eval[2]
 *   evec[row*3 + col] is the component (row,col), with eigenvectors
 *   stored in columns.
 *
 * The algorithm follows the robust "most-distinct eigenpair + 2x2
 * deflation" strategy used for 3x3 real symmetric matrices.  It does not
 * call BLAS, LAPACK, OpenMP, CUDA, or any external numerical library.
 */
void qdiag_solve_q3(
    double qxx,
    double qyy,
    double qxy,
    double qxz,
    double qyz,
    double eval[3],
    double evec[9]
);

/* Compute only the largest eigenpair, avoiding the deflated 2x2 solve. */
void qdiag_dominant_q3(
    double qxx,
    double qyy,
    double qxy,
    double qxz,
    double qyz,
    double *eval,
    double evec[3]
);

#ifdef __cplusplus
}
#endif

#endif
