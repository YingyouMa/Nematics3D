#ifndef LBM_AN_ACTIVE_NEMATIC_H_
#define LBM_AN_ACTIVE_NEMATIC_H_

#include <memory>
#include <string>
#include <iostream>

#include "grid.h"
#include "params.h"
#include "fluid_fields.h"
#include "qtensor_fields.h"
#include "lbm_solver.h"
// #include "qtensor_solver.h"
#include "sim_io.h"
// Orchestrates LbmSolver + QTensorSolver + SimIO for 2D active nematics.
//
// To use a custom activity model, inject a QTensorSolver subclass:
//
//   ActiveNematicSim<PeriodicBC> sim{grid, std::make_unique<VaryingAlpha>(grid)};
//
// To run without any Q-tensor dynamics use LbmSolver directly.
template<typename BC>
class PoiseuilleSim {
    FluidFields    fluid_;
    QTensorFields  qtensor_;
    LbmSolver<BC>  lbm_;
    SimIO          io_;
    int            time_step_ = 0;
    int            num_files_exported = 0;

    void Initialize() {
        lbm_.Initialize(fluid_);
        for (int x : std::views::iota(0, nx)) {
            for (int y : std::views::iota(0, ny)) {
                for (int z : std::views::iota(0, nz)) {
                    fluid_.fx[x, y, z] = kDeltaP;
                    fluid_.fy[x, y, z] = 0.0;
                    fluid_.fz[x, y, z] = 0.0;
                }
            }
        }

    }

public:
    // Default: constant-alpha active nematic.
    // Supply a QTensorSolver subclass to override the activity model.
    explicit PoiseuilleSim(Grid<BC> grid)
        : lbm_(grid)
    {
        Initialize();
        io_.LogSetupSummary(Grid<BC>::GridType());
    }

    // Q-tensor FD step + active force + LBM step.
    void Step() {
        lbm_.LatticeBoltzmannStep(fluid_);
        ++time_step_;
    }

    // Returns false if the simulation has diverged (NaN detected).
    bool Log() { return io_.Log(fluid_, time_step_); }

    void Export(const std::string& path) {
        // io_.Export(fluid_, qtensor_, path, time_step_);
        io_.ExportVTKHDF(fluid_, qtensor_, path, num_files_exported, static_cast<double>(time_step_)*Params::DT);
        num_files_exported++;
    }

    void ExportDistribution(const std::string& path) {
        io_.ExportDistribution(fluid_, path, time_step_);
    }

    int GetTimeStep() const { return time_step_; }
};

#endif // LBM_AN_ACTIVE_NEMATIC_H_
