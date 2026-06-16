#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/flows/GradientFlow.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 16, 16, 1.0, 1.0 };

    O3SigmaModel model{ 1.0 };
    GradientFlow flow{ 0.05 };

    // Uniform field should remain uniform and have zero energy.
    O3Field uniform = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });

    flow.run(uniform, model, 10, 1);

    const double uniform_energy = model.total_energy(uniform);

    assert(std::abs(uniform_energy) < 1e-12);

    for (std::size_t k = 0; k < uniform.size(); ++k) {
        const Vec3& v = uniform.at_index(k);

        assert(std::abs(v.x - 0.0) < 1e-12);
        assert(std::abs(v.y - 0.0) < 1e-12);
        assert(std::abs(v.z - 1.0) < 1e-12);
        assert(std::abs(v.norm() - 1.0) < 1e-12);
    }

    // A localized perturbation should relax and reduce energy.
    O3Field field = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });
    field(8, 8) = Vec3{ 1.0, 0.0, 0.0 };
    field.normalize_all();

    const double energy_before = model.total_energy(field);

    auto history = flow.run(field, model, 20, 5);

    const double energy_after = model.total_energy(field);

    assert(std::isfinite(energy_before));
    assert(std::isfinite(energy_after));
    assert(energy_before > 0.0);
    assert(energy_after < energy_before);

    assert(history.size() == 5);

    for (const auto& record : history) {
        assert(std::isfinite(record.energy));
        assert(std::isfinite(record.topological_charge));
    }

    for (std::size_t k = 0; k < field.size(); ++k) {
        assert(std::abs(field.at_index(k).norm() - 1.0) < 1e-12);
    }

    // Fixed boundaries stay pinned, while Neumann boundaries can evolve.
    Lattice2D fixed_lat{
        8,
        8,
        1.0,
        1.0,
        BoundaryCondition::Fixed
    };
    O3Field fixed = O3Field::uniform(fixed_lat, Vec3{ 0.0, 0.0, 1.0 });
    fixed(0, 4) = Vec3{ 1.0, 0.0, 0.0 };
    const Vec3 fixed_before = fixed(0, 4);
    flow.step(fixed);

    assert(std::abs(fixed(0, 4).x - fixed_before.x) < 1e-12);
    assert(std::abs(fixed(0, 4).z - fixed_before.z) < 1e-12);

    Lattice2D neumann_lat{
        8,
        8,
        1.0,
        1.0,
        BoundaryCondition::Neumann
    };
    O3Field neumann = O3Field::uniform(neumann_lat, Vec3{ 0.0, 0.0, 1.0 });
    neumann(0, 4) = Vec3{ 1.0, 0.0, 0.0 };
    flow.step(neumann);

    assert(neumann(0, 4).x < 1.0);
    assert(neumann(0, 4).z > 0.0);

    Lattice2D dirichlet_lat{
        8,
        8,
        1.0,
        1.0,
        BoundaryCondition::Dirichlet
    };
    O3Field dirichlet = O3Field::uniform(
        dirichlet_lat,
        Vec3{ 1.0, 0.0, 0.0 }
    );

    assert(std::abs(dirichlet(0, 4).x) < 1e-12);
    assert(std::abs(dirichlet(0, 4).y) < 1e-12);
    assert(std::abs(dirichlet(0, 4).z - 1.0) < 1e-12);

    dirichlet(0, 4) = Vec3{ 1.0, 0.0, 0.0 };
    flow.step(dirichlet);

    assert(std::abs(dirichlet(0, 4).x) < 1e-12);
    assert(std::abs(dirichlet(0, 4).y) < 1e-12);
    assert(std::abs(dirichlet(0, 4).z - 1.0) < 1e-12);

    std::cout << "GradientFlow tests passed successfully\n";
    std::cout << "Energy before = " << energy_before << "\n";
    std::cout << "Energy after  = " << energy_after << "\n";

    return 0;
}
