#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 8, 8, 1.0, 1.0 };

    O3SigmaModel model{ 1.0 };

    // Uniform field should have zero energy.
    O3Field uniform = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });

    const double uniform_energy = model.total_energy(uniform);

    assert(std::abs(uniform_energy) < 1e-12);

    // A non-uniform field should have positive energy.
    O3Field field = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });

    field(0, 0) = Vec3{ 1.0, 0.0, 0.0 };
    field.normalize_all();

    const double energy = model.total_energy(field);

    assert(energy > 0.0);

    // Energy density should be non-negative everywhere.
    for (std::size_t j = 0; j < lat.ny(); ++j) {
        for (std::size_t i = 0; i < lat.nx(); ++i) {
            assert(model.energy_density_at(field, i, j) >= 0.0);
        }
    }

    std::cout << "O3SigmaModel tests passed successfully\n";

    return 0;
}