#include <cassert>
#include <cmath>
#include <iostream>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/flows/BabySkyrmeGradientFlow.hpp"
#include "solitonkit/initializers/SkyrmionAnsatz.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"

using namespace solitonkit;

int main() {
    Lattice2D lat{ 32, 32, 0.5, 0.5 };
    BabySkyrmeModel model{ 1.0, 1.0 };

    O3Field vacuum = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });
    const BabySkyrmeEnergyTerms vacuum_terms = model.energy_terms(vacuum);

    assert(std::abs(vacuum_terms.sigma) < 1e-12);
    assert(std::abs(vacuum_terms.skyrme) < 1e-12);
    assert(std::abs(vacuum_terms.potential) < 1e-12);
    assert(std::abs(vacuum_terms.total()) < 1e-12);

    O3Field field = SkyrmionAnsatz::charge_one(lat, 3.0);
    const BabySkyrmeEnergyTerms terms = model.energy_terms(field);

    assert(terms.sigma > 0.0);
    assert(terms.skyrme > 0.0);
    assert(terms.potential > 0.0);
    assert(std::abs(terms.total() - model.energy(field)) < 1e-12);

    BabySkyrmeGradientFlow flow{ 1e-4 };
    const double energy_before = model.energy(field);
    const auto history = flow.run(field, model, 10, 2);
    const double energy_after = model.energy(field);

    assert(energy_after < energy_before);
    assert(history.size() == 6);
    assert(std::abs(history.front().energy - energy_before) < 1e-12);
    assert(std::abs(history.back().energy - energy_after) < 1e-12);

    for (const auto& record : history) {
        assert(std::isfinite(record.energy));
        assert(std::isfinite(record.topological_charge));
    }

    for (std::size_t k = 0; k < field.size(); ++k) {
        assert(std::abs(field.at_index(k).norm() - 1.0) < 1e-12);
    }

    std::cout << "BabySkyrmeModel and BabySkyrmeGradientFlow tests passed\n";
    std::cout << "Energy before = " << energy_before << "\n";
    std::cout << "Energy after  = " << energy_after << "\n";

    return 0;
}
