#ifndef LBM_AN_PARAMS_H_
#define LBM_AN_PARAMS_H_

#include <cmath> // for sqrt

namespace Params {

    // Speed of sound related constants

    static constexpr double kCs2Inv = 3.0; // 1/c_s^2
    static constexpr double kCs2InvTimes2 = 6.0; // 2/c_s^2
    static constexpr double kCs4Inv = 9.0; // 1/c_s^4
    static constexpr double khalfCs2Inv = 1.5; // 1/2 * 1/c_s^2
    static constexpr double khalfCs4Inv = 4.5; // 1/2 * 1/c_s^4
    // Grid
    inline constexpr int nx = 64;
    inline constexpr int ny = 32;
    inline constexpr int nz = 32;
    inline constexpr int ndir = 15;
    inline constexpr int nq = 3;
    inline constexpr int numprocs = 1;

    // Spatial / temporal
    inline constexpr double DX = 1.0, DY = 1.0, DZ = 1.0;
    inline constexpr double DT = 1.0;

    // LBM relaxation
    inline constexpr double RHO = 1.1;        // initial lattice density
    inline constexpr double kDensity = 0.1;   // physical density scale
    inline constexpr double TAUF = std::sqrt(static_cast<double>(3.0f/16.0f))+0.5; // Relaxation time;
    inline constexpr double nu = (2 * TAUF - 1) / 6.0f; // kinematic shear viscosity
    inline constexpr double u_max = 0.1;
    inline constexpr double kDeltaP = 8 * nu * u_max / static_cast<double>(ny * ny * nz);


    inline constexpr double omega         = 1.0 - DT / TAUF;
    inline constexpr double omega_prime   = DT / TAUF;
    inline constexpr double omega_forcing = 1.0 - DT / 2.0 / TAUF;

    // Free-energy / elasticity
    inline constexpr double L = 0.1;                          // Frank elasticity
    inline constexpr double A = 1.0 - RHO;
    inline constexpr double B = -1.0 * 0.1 * 3.5;
    inline constexpr double C = (1.0 + RHO) / (RHO * RHO);

    // Q-tensor dynamics
    inline constexpr double LAMBDA = 1.0;    // flow-aligning
    inline constexpr double GAMMA  = 0.1;    // inverse rotational viscosity

    // Activity & friction
    inline constexpr double ALPHA = 0.2;
    inline constexpr double MU    = 0.0;

    // Initial conditions
    inline constexpr double NOISE = 0.05;

    // Wall BC (used by HandleBoundaries)
    inline constexpr double kLidVelocity = 0.1;

    // Derived physical units
    inline constexpr double PHYSICAL_DT    = DT * DX * DY / (GAMMA * L);
    inline constexpr double PHYSICAL_RHO   = kDensity * DT * DT / (GAMMA * GAMMA * L);
    inline constexpr double PHYSICAL_NU    = kDensity * DT / 3.0 * (TAUF / DT - 0.5);
    inline constexpr double PHYSICAL_ALPHA = ALPHA * L / (DX * DX);

    // Logging verbosity
    inline constexpr bool kDebugLogging = false;
}

#endif // LBM_AN_PARAMS_H_
