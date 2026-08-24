#include "qdiag_kernel.h"

#include <math.h>

#ifndef QDIAG_PI
#define QDIAG_PI 3.141592653589793238462643383279502884
#endif

static void qdiag_normalize_best_adjugate_row(
    double a,
    double d,
    double b,
    double c,
    double e,
    double lambda,
    double *vx,
    double *vy,
    double *vz
)
{
    const double r00 = a - lambda;
    const double r11 = d - lambda;
    const double r22 = -a - d - lambda;
    const double A00 = r11*r22 - e*e;
    const double A11 = r00*r22 - c*c;
    const double A22 = r00*r11 - b*b;
    const double A01 = c*e - b*r22;
    const double A02 = b*e - c*r11;
    const double A12 = b*c - r00*e;
    const double n0 = A00*A00 + A01*A01 + A02*A02;
    const double n1 = A01*A01 + A11*A11 + A12*A12;
    const double n2 = A02*A02 + A12*A12 + A22*A22;
    double norm2;

    if (n0 >= n1 && n0 >= n2) {
        *vx = A00; *vy = A01; *vz = A02; norm2 = n0;
    } else if (n1 >= n2) {
        *vx = A01; *vy = A11; *vz = A12; norm2 = n1;
    } else {
        *vx = A02; *vy = A12; *vz = A22; norm2 = n2;
    }

    if (!(norm2 > 0.0) || !isfinite(norm2)) {
        *vx = 1.0; *vy = 0.0; *vz = 0.0;
    } else {
        const double inv_norm = 1.0 / sqrt(norm2);
        *vx *= inv_norm; *vy *= inv_norm; *vz *= inv_norm;
    }
}

void qdiag_dominant_q3(
    double a,
    double d,
    double b,
    double c,
    double e,
    double *eval,
    double evec[3]
)
{
    if (a == 0.0 && d == 0.0 && b == 0.0 && c == 0.0 && e == 0.0) {
        *eval = 0.0;
        evec[0] = 1.0; evec[1] = 0.0; evec[2] = 0.0;
        return;
    }

    const double p2 = (a*a + d*d + a*d + b*b + c*c + e*e) / 3.0;
    const double p = sqrt(p2);
    const double detq =
        -a*d*(a + d) + 2.0*b*c*e - a*e*e - d*c*c + (a + d)*b*b;
    double r = 0.5 * detq / (p*p*p);
    if (r > 1.0) {
        r = 1.0;
    } else if (r < -1.0) {
        r = -1.0;
    }

    *eval = 2.0*p*cos(acos(r) / 3.0);
    qdiag_normalize_best_adjugate_row(
        a, d, b, c, e, *eval, &evec[0], &evec[1], &evec[2]
    );
}

void qdiag_solve_q3(
    double a,
    double d,
    double b,
    double c,
    double e,
    double eval[3],
    double evec[9]
)
{
    /*
     * Q = [[a,b,c],[b,d,e],[c,e,-a-d]].
     *
     * The numerical path intentionally exploits exact tracelessness and
     * bounded physical Q values.  No per-tensor scale normalization is done.
     */
    if (a == 0.0 && d == 0.0 && b == 0.0 && c == 0.0 && e == 0.0) {
        eval[0] = eval[1] = eval[2] = 0.0;
        evec[0] = 1.0; evec[1] = 0.0; evec[2] = 0.0;
        evec[3] = 0.0; evec[4] = 1.0; evec[5] = 0.0;
        evec[6] = 0.0; evec[7] = 0.0; evec[8] = 1.0;
        return;
    }

    /*
     * For trace(Q)=0:
     *   p^2 = tr(Q^2)/6
     *       = (a^2 + d^2 + a*d + b^2 + c^2 + e^2)/3.
     */
    const double p2 = (a*a + d*d + a*d + b*b + c*c + e*e) / 3.0;
    const double p = sqrt(p2);

    /*
     * det(Q) with qzz eliminated using qzz = -a-d.
     */
    const double detq =
        -a*d*(a + d)
        + 2.0*b*c*e
        - a*e*e
        - d*c*c
        + (a + d)*b*b;

    /*
     * Trigonometric cubic, but evaluate only the most-distinct root.
     */
    double r = 0.5 * detq / (p*p*p);
    if (r > 1.0) {
        r = 1.0;
    } else if (r < -1.0) {
        r = -1.0;
    }

    const double phi = acos(r) / 3.0;
    const int upper = (r >= 0.0);
    const double lambda_distinct = upper
        ? 2.0*p*cos(phi)
        : 2.0*p*cos(phi + 2.0*QDIAG_PI/3.0);

    /*
     * R = Q - lambda_distinct I.
     */
    double vx;
    double vy;
    double vz;
    qdiag_normalize_best_adjugate_row(
        a, d, b, c, e, lambda_distinct, &vx, &vy, &vz
    );

    /*
     * Stable orthonormal basis {u,w} for v^perp.  Choose the Cartesian axis
     * least aligned with v, then project it into the perpendicular plane.
     */
    const double avx = fabs(vx);
    const double avy = fabs(vy);
    const double avz = fabs(vz);

    double ex = 0.0;
    double ey = 0.0;
    double ez = 0.0;
    double dot;

    if (avx <= avy && avx <= avz) {
        ex = 1.0;
        dot = vx;
    } else if (avy <= avz) {
        ey = 1.0;
        dot = vy;
    } else {
        ez = 1.0;
        dot = vz;
    }

    double un2 = 1.0 - dot*dot;
    if (un2 < 0.0) {
        un2 = 0.0;
    }

    const double inv_un = 1.0 / sqrt(un2);
    const double ux = (ex - dot*vx) * inv_un;
    const double uy = (ey - dot*vy) * inv_un;
    const double uz = (ez - dot*vz) * inv_un;

    const double wx = vy*uz - vz*uy;
    const double wy = vz*ux - vx*uz;
    const double wz = vx*uy - vy*ux;

    /*
     * Restrict Q to span(u,w).  Exact tracelessness lets us avoid a full
     * second projected diagonal entry:
     *
     *   C2 = -lambda_distinct - A2.
     */
    const double A2 =
        a*ux*ux
        + d*uy*uy
        - (a + d)*uz*uz
        + 2.0*(b*ux*uy + c*ux*uz + e*uy*uz);

    const double B2 =
        wx*(a*ux + b*uy + c*uz)
        + wy*(b*ux + d*uy + e*uz)
        + wz*(c*ux + e*uy - (a + d)*uz);

    const double diff2 = 2.0*A2 + lambda_distinct;
    /* Physical Q components are bounded, so direct squaring cannot overflow. */
    const double disc2 = sqrt(diff2*diff2 + 4.0*B2*B2);

    /*
     * Stable remaining eigenvalues.  The sign of r identifies whether the
     * isolated root is the largest or smallest one.
     */
    const double far = upper
        ? 0.5*(-lambda_distinct - disc2)
        : 0.5*(-lambda_distinct + disc2);

    const double near = detq / (lambda_distinct * far);

    const double lambda_lo = upper ? far : near;
    const double lambda_hi = upper ? near : far;

    /*
     * Stable algebraic Jacobi rotation for the larger eigenvalue of the
     * deflated 2x2 block.  No atan2/sin/cos are needed here.
     */
    double ct;
    double st;

    if (disc2 == 0.0) {
        ct = 1.0;
        st = 0.0;
    } else if (diff2 >= 0.0) {
        const double k = 2.0*B2 / (disc2 + diff2);
        const double inv = 1.0 / sqrt(1.0 + k*k);
        ct = inv;
        st = k*inv;
    } else {
        const double k = 2.0*B2 / (disc2 - diff2);
        const double inv = 1.0 / sqrt(1.0 + k*k);
        ct = fabs(k)*inv;
        st = (B2 >= 0.0) ? inv : -inv;
    }

    const double vhx = ct*ux + st*wx;
    const double vhy = ct*uy + st*wy;
    const double vhz = ct*uz + st*wz;

    const double vlx = -st*ux + ct*wx;
    const double vly = -st*uy + ct*wy;
    const double vlz = -st*uz + ct*wz;

    if (upper) {
        eval[0] = lambda_lo;
        eval[1] = lambda_hi;
        eval[2] = lambda_distinct;

        evec[0] = vlx; evec[1] = vhx; evec[2] = vx;
        evec[3] = vly; evec[4] = vhy; evec[5] = vy;
        evec[6] = vlz; evec[7] = vhz; evec[8] = vz;
    } else {
        eval[0] = lambda_distinct;
        eval[1] = lambda_lo;
        eval[2] = lambda_hi;

        evec[0] = vx;  evec[1] = vlx; evec[2] = vhx;
        evec[3] = vy;  evec[4] = vly; evec[5] = vhy;
        evec[6] = vz;  evec[7] = vlz; evec[8] = vhz;
    }
}
