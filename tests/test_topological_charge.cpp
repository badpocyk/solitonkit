#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 16, 16, 1.0, 1.0 };

    // Uniform field should have zero topological charge.
    O3Field uniform = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });

    const double q_uniform = TopologicalCharge::total(uniform);

    assert(std::abs(q_uniform) < 1e-12);

    // Random field should produce a finite number.
    O3Field random_field = O3Field::random(lat, 42);

    const double q_random = TopologicalCharge::total(random_field);

    assert(std::isfinite(q_random));

    // Density should also be finite everywhere.
    for (std::size_t j = 0; j < lat.ny(); ++j) {
        for (std::size_t i = 0; i < lat.nx(); ++i) {
            const double density = TopologicalCharge::density_at(random_field, i, j);
            assert(std::isfinite(density));
        }
    }

    std::cout << "TopologicalCharge tests passed successfully\n";

    return 0;
}