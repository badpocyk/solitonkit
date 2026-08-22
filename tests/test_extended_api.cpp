#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/analysis/LinearStability.hpp"
#include "solitonkit/analysis/Continuation.hpp"
#include "solitonkit/core/BoundaryConditions.hpp"
#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/Lattice3D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/ScalarField2D.hpp"
#include "solitonkit/dynamics/LLGDynamics.hpp"
#include "solitonkit/initializers/HopfionAnsatz.hpp"
#include "solitonkit/models/HopfionModel.hpp"
#include "solitonkit/models/MicromagneticModel.hpp"
#include "solitonkit/models/ScalarModels.hpp"
#include "solitonkit/operators/DifferentialOperators.hpp"
#include "solitonkit/solvers/Solvers.hpp"
#include "solitonkit/solvers/StationarySolvers.hpp"
#include "solitonkit/topology/Topology.hpp"

using namespace solitonkit;

int main() {
    const BoundaryConditions2D mixed{
        BoundaryCondition::Periodic,
        BoundaryCondition::Dirichlet
    };
    const Lattice2D mixed_lattice{ 7, 5, 0.5, 0.25, mixed };
    assert(mixed_lattice.left(0) == 6);
    assert(mixed_lattice.down(0) == 0);
    assert(!mixed_lattice.is_fixed_boundary(0, 2));
    assert(mixed_lattice.is_fixed_boundary(3, 0));
    O3Field mixed_field{ mixed_lattice, Vec3{ 1.0, 0.0, 0.0 } };
    assert(std::abs(mixed_field(0, 2).x - 1.0) < 1e-12);
    assert(std::abs(mixed_field(3, 0).z - 1.0) < 1e-12);

    const Lattice2D fixed_lattice{
        9, 7, 0.25, 0.5, BoundaryCondition::Fixed
    };
    ScalarField2D scalar{ fixed_lattice };
    for (std::size_t j = 0; j < fixed_lattice.ny(); ++j) {
        for (std::size_t i = 0; i < fixed_lattice.nx(); ++i) {
            scalar(i, j) = 2.0 * static_cast<double>(i) * fixed_lattice.dx()
                - 3.0 * static_cast<double>(j) * fixed_lattice.dy();
        }
    }
    assert(std::abs(differential::derivative_x(scalar, 4, 3) - 2.0) < 1e-12);
    assert(std::abs(differential::derivative_y(scalar, 4, 3) + 3.0) < 1e-12);
    assert(std::abs(differential::laplacian(scalar, 4, 3)) < 1e-12);

    const Lattice2D neumann_lattice{
        6, 6, 1.0, 1.0, BoundaryCondition::Neumann
    };
    ScalarField2D neumann{ neumann_lattice };
    neumann(1, 3) = 2.0;
    assert(std::abs(differential::derivative_x(neumann, 0, 3)) < 1e-12);

    const Lattice2D model_lattice{
        12, 12, 0.5, 0.5, BoundaryCondition::Dirichlet
    };
    ScalarField2D phi{ model_lattice, 1.0, 1.0 };
    Phi4Model phi4{ 1.0, 1.0 };
    Model* model_base = &phi4;
    assert(model_base->name() == "phi4");
    assert(std::abs(phi4.energy(phi)) < 1e-12);
    phi(6, 6) = 0.0;
    const double phi_before = phi4.energy(phi);
    MinimizeOptions minimize_options;
    minimize_options.max_steps = 8;
    minimize_options.step_size = 0.05;
    minimize_options.record_every = 2;
    const auto phi_history = minimize(phi, phi4, minimize_options);
    assert(phi4.energy(phi) < phi_before);
    assert(phi_history.size() >= 2);

    ScalarField2D unstable_phi{ Lattice2D{ 4, 4 }, 0.0 };
    StabilityOptions stability_options;
    stability_options.modes = 1;
    stability_options.max_iterations = 40;
    stability_options.subspace_dimension = 16;
    stability_options.tolerance = 1e-6;
    const auto unstable_stability = stability_analysis(
        unstable_phi, phi4, stability_options
    );
    assert(unstable_stability.stationary);
    assert(unstable_stability.converged);
    assert(!unstable_stability.stable);
    assert(unstable_stability.eigenvalues.size() == 1);
    assert(std::abs(unstable_stability.eigenvalues[0] + 1.0) < 1e-5);
    assert(unstable_stability.modes[0].size() == unstable_phi.size());

    std::vector<double> constant_mode(unstable_phi.size(), 1.0);
    const auto hessian_mode = hessian_vector_product(
        unstable_phi, phi4, constant_mode
    );
    for (const double value : hessian_mode) {
        assert(std::abs(value + 1.0) < 1e-5);
    }

    ScalarField2D stable_phi{
        Lattice2D{ 4, 4, 1.0, 1.0, BoundaryCondition::Dirichlet },
        1.0,
        1.0
    };
    stability_options.subspace_dimension = 4;
    const auto stable_stability = stability_analysis(
        stable_phi, phi4, stability_options
    );
    assert(stable_stability.degrees_of_freedom == 4);
    assert(stable_stability.stable);
    for (std::size_t i = 0; i < 4; ++i) {
        assert(std::abs(stable_stability.modes[0][i]) < 1e-12);
        assert(std::abs(stable_stability.modes[0][12 + i]) < 1e-12);
    }

    ScalarField2D newton_phi{ Lattice2D{ 4, 4 }, 0.8 };
    StationaryOptions stationary_options;
    stationary_options.max_steps = 10;
    stationary_options.tolerance = 1e-9;
    stationary_options.preconditioner_probes = 2;
    stationary_options.gmres.tolerance = 1e-9;
    const auto stationary_history = solve_stationary(
        newton_phi, phi4, stationary_options
    );
    assert(stationary_history.back().converged);
    assert(stationary_history.back().residual_norm < 1e-9);
    assert(std::abs(newton_phi(0, 0) - 1.0) < 1e-8);

    ContinuationOptions continuation_options;
    continuation_options.start = 0.8;
    continuation_options.stop = 1.1;
    continuation_options.step = 0.1;
    continuation_options.maximum_step = 0.15;
    continuation_options.max_points = 12;
    continuation_options.corrector_tolerance = 1e-8;
    continuation_options.stationary.tolerance = 1e-9;
    continuation_options.stationary.max_steps = 10;
    continuation_options.stationary.preconditioner_probes = 1;
    continuation_options.stationary.gmres.tolerance = 1e-8;
    continuation_options.preconditioner_probes = 1;
    continuation_options.gmres.tolerance = 1e-7;
    continuation_options.stability.modes = 1;
    continuation_options.stability.subspace_dimension = 9;
    continuation_options.stability.tolerance = 1e-6;
    const ScalarField2D continuation_initial{ Lattice2D{ 3, 3 }, 0.8 };
    const auto branch = continue_solution(
        continuation_initial,
        [](double vacuum) { return Phi4Model{ 1.0, vacuum }; },
        continuation_options
    );
    assert(branch.converged);
    assert(branch.reached_stop);
    assert(branch.points.size() >= 3);
    for (const auto& point : branch.points) {
        assert(point.converged);
        assert(std::abs(point.field.at_index(0) - point.parameter) < 1e-6);
        assert(point.lowest_eigenvalue > 0.0);
    }

    ScalarField2D sine_field{ model_lattice, 0.0, 0.0 };
    SineGordonModel sine_gordon{ 1.0, 1.0 };
    assert(std::abs(sine_gordon.energy(sine_field)) < 1e-12);

    XYField xy{ Lattice2D{ 8, 8 }, 0.25 };
    XYModel xy_model{ 1.0, 0.0 };
    assert(std::abs(xy_model.energy(xy)) < 1e-12);
    xy(4, 4) = 1.0;
    const double xy_before = xy_model.energy(xy);
    SolveOptions solve_options;
    solve_options.steps = 5;
    solve_options.time_step = 0.02;
    solve_options.record_every = 1;
    const auto xy_history = solve(xy, xy_model, solve_options);
    assert(xy_model.energy(xy) < xy_before);
    assert(xy_history.size() == 6);

    XYField vortex{
        Lattice2D{ 6, 6, 1.0, 1.0, BoundaryCondition::Neumann }
    };
    for (std::size_t j = 0; j < 6; ++j) {
        for (std::size_t i = 0; i < 6; ++i) {
            vortex(i, j) = std::atan2(
                static_cast<double>(j) - 2.5,
                static_cast<double>(i) - 2.5
            );
        }
    }
    const auto defects = topology::detect_defects(vortex);
    assert(defects.size() == 1);
    assert(defects[0].charge == 1);
    assert(topology::vortex_number(vortex) == 1);

    const Lattice2D magnetic_lattice{
        12, 12, 0.5, 0.5, BoundaryCondition::Periodic
    };
    O3Field magnetization{ magnetic_lattice, Vec3{ 0.0, 0.0, 1.0 } };
    MicromagneticModel micromagnetic{
        1.0,
        0.1,
        0.2,
        Vec3{ 0.0, 0.0, 0.5 },
        Vec3{ 0.0, 0.0, 1.0 },
        DMIType::Interfacial
    };
    const auto vacuum_terms = micromagnetic.energy_terms(magnetization);
    assert(std::abs(vacuum_terms.exchange) < 1e-12);
    assert(std::abs(vacuum_terms.dmi) < 1e-12);
    assert(std::abs(vacuum_terms.anisotropy) < 1e-12);
    magnetization(6, 6) = Vec3{ 1.0, 0.0, 0.0 };
    const double magnetic_before = micromagnetic.energy(magnetization);
    LLGDynamics llg{ 1e-3, 0.5 };
    const auto llg_history = llg.run(magnetization, micromagnetic, 10, 5);
    assert(micromagnetic.energy(magnetization) < magnetic_before);
    assert(llg_history.size() == 3);
    for (std::size_t index = 0; index < magnetization.size(); ++index) {
        assert(std::abs(magnetization.at_index(index).norm() - 1.0) < 1e-12);
    }

    const Lattice3D lattice3d{
        9, 9, 9, 0.5, 0.5, 0.5, BoundaryCondition::Dirichlet
    };
    assert(lattice3d.index(1, 0, 0) == 1);
    assert(lattice3d.index(0, 1, 0) == 9);
    assert(lattice3d.index(0, 0, 1) == 81);
    const HopfionSpec hopfion_spec{ 1.5, 1, 1 };
    assert(hopfion_spec.hopf_charge() == 1);
    O3Field3D hopfion = HopfionAnsatz::create(lattice3d, hopfion_spec);
    for (std::size_t index = 0; index < hopfion.size(); ++index) {
        assert(std::abs(hopfion.at_index(index).norm() - 1.0) < 1e-12);
    }
    assert(std::abs(hopfion(0, 4, 4).z - 1.0) < 1e-12);
    HopfionModel hopfion_model{ 1.0, 0.25, 0.1 };
    const auto hopfion_terms = hopfion_model.energy_terms(hopfion);
    assert(hopfion_terms.sigma > 0.0);
    assert(hopfion_terms.skyrme > 0.0);
    assert(hopfion_terms.potential > 0.0);

    O3Field3D uniform_3d{ lattice3d, Vec3{ 0.0, 0.0, 1.0 } };
    const auto uniform_hopf_charge = topology::hopf_charge(uniform_3d);
    assert(uniform_hopf_charge.converged);
    assert(std::abs(uniform_hopf_charge.charge) < 1e-12);

    const Lattice3D topology_lattice{
        17, 17, 17, 0.4, 0.4, 0.4, BoundaryCondition::Dirichlet
    };
    const O3Field3D topology_hopfion = HopfionAnsatz::create(
        topology_lattice, HopfionSpec{ 1.5, 1, 1 }
    );
    const auto numerical_hopf_charge = topology::hopf_charge(
        topology_hopfion, topology::HopfChargeOptions{ 1000, 1e-7 }
    );
    assert(numerical_hopf_charge.converged);
    assert(std::abs(numerical_hopf_charge.charge) > 0.4);
    assert(std::abs(numerical_hopf_charge.charge) < 1.2);

    MinimizeOptions hopf_options;
    hopf_options.max_steps = 1;
    hopf_options.step_size = 1e-3;
    hopf_options.record_every = 1;
    const double hopf_before = hopfion_model.energy(hopfion);
    minimize(hopfion, hopfion_model, hopf_options);
    assert(hopfion_model.energy(hopfion) <= hopf_before + 1e-12);

    std::cout << "Extended API tests passed\n";
    return 0;
}
