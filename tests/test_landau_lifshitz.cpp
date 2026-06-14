#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/dynamics/LandauLifshitzDynamics.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 32, 32, 0.5, 0.5 };
    BabySkyrmeModel model{ 1.0, 1.0 };

    LandauLifshitzDynamics conservative{ 1e-5, 0.0 };
    O3Field vacuum = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });
    conservative.run(vacuum, model, 10, 2);

    for (std::size_t k = 0; k < vacuum.size(); ++k) {
        assert(std::abs(vacuum.at_index(k).x) < 1e-12);
        assert(std::abs(vacuum.at_index(k).y) < 1e-12);
        assert(std::abs(vacuum.at_index(k).z - 1.0) < 1e-12);
    }

    O3Field precessing = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });
    precessing(16, 16) = Vec3{ 1.0, 0.0, 0.0 };
    const Vec3 before = precessing(16, 16);
    conservative.step(precessing, model);

    assert(std::abs(precessing(16, 16).y - before.y) > 1e-8);

    for (std::size_t k = 0; k < precessing.size(); ++k) {
        assert(std::abs(precessing.at_index(k).norm() - 1.0) < 1e-12);
    }

    O3Field damped_field = SkyrmionAnsatz::charge_one(lat, 3.0);
    LandauLifshitzDynamics damped{ 1e-5, 0.5 };
    const double energy_before = model.energy(damped_field);
    const auto history = damped.run(damped_field, model, 20, 5);
    const double energy_after = model.energy(damped_field);

    assert(energy_after < energy_before);
    assert(history.size() == 5);
    assert(std::abs(history.back().time - 20e-5) < 1e-12);

    Lattice2D fixed_lat{
        16,
        16,
        0.5,
        0.5,
        BoundaryCondition::Fixed
    };
    O3Field fixed = SkyrmionAnsatz::charge_one(fixed_lat, 2.0);
    const Vec3 fixed_before = fixed(0, 8);
    damped.step(fixed, model);

    assert(std::abs(fixed(0, 8).x - fixed_before.x) < 1e-12);
    assert(std::abs(fixed(0, 8).y - fixed_before.y) < 1e-12);
    assert(std::abs(fixed(0, 8).z - fixed_before.z) < 1e-12);

    std::cout << "LandauLifshitzDynamics tests passed\n";
    std::cout << "Damped energy before = " << energy_before << "\n";
    std::cout << "Damped energy after  = " << energy_after << "\n";

    return 0;
}
