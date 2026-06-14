#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 64, 64, 1.0, 1.0 };

    O3Field skyrmion = SkyrmionAnsatz::charge_one(lat, 8.0);

    // Field should stay on S^2.
    for (std::size_t k = 0; k < skyrmion.size(); ++k) {
        const double n = skyrmion.at_index(k).norm();
        assert(std::abs(n - 1.0) < 1e-12);
    }

    // Topological charge should be close to +1.
    // It is not exactly 1 because we currently use a simple finite-difference formula.
    const double q = TopologicalCharge::total(skyrmion);

    assert(std::isfinite(q));
    assert(std::abs(q - 1.0) < 0.15);

    // Energy should be finite and positive.
    O3SigmaModel model{ 1.0 };
    const double energy = model.total_energy(skyrmion);

    assert(std::isfinite(energy));
    assert(energy > 0.0);

    O3Field anti_skyrmion = SkyrmionAnsatz::anti_charge_one(lat, 8.0);
    const double q_anti = TopologicalCharge::total(anti_skyrmion);

    assert(std::isfinite(q_anti));
    assert(std::abs(q_anti + 1.0) < 0.15);

    std::cout << "SkyrmionAnsatz tests passed successfully\n";
    std::cout << "Q skyrmion      = " << q << "\n";
    std::cout << "Q anti-skyrmion = " << q_anti << "\n";
    std::cout << "Energy          = " << energy << "\n";

    return 0;
}