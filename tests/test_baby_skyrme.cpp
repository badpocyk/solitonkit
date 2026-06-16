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
    assert(std::abs(vacuum_terms.dmi) < 1e-12);
    assert(std::abs(vacuum_terms.total()) < 1e-12);

    O3Field field = SkyrmionAnsatz::charge_one(lat, 3.0);
    const BabySkyrmeEnergyTerms terms = model.energy_terms(field);

    assert(terms.sigma > 0.0);
    assert(terms.skyrme > 0.0);
    assert(terms.potential > 0.0);
    assert(std::abs(terms.dmi) < 1e-12);
    assert(std::abs(terms.total() - model.energy(field)) < 1e-12);

    constexpr double pi = 3.141592653589793238462643383279502884;
    Lattice2D dmi_lat{ 32, 8, 0.25, 0.5 };
    O3Field helix{ dmi_lat };
    const double wave_number =
        2.0 * pi / (static_cast<double>(dmi_lat.nx()) * dmi_lat.dx());

    for (std::size_t j = 0; j < dmi_lat.ny(); ++j) {
        for (std::size_t i = 0; i < dmi_lat.nx(); ++i) {
            const double x = static_cast<double>(i) * dmi_lat.dx();
            helix(i, j) = Vec3{
                0.0,
                std::sin(wave_number * x),
                std::cos(wave_number * x)
            };
        }
    }

    const double dmi_strength = 0.75;
    BabySkyrmeModel dmi_model{ 0.0, 0.0, dmi_strength };
    const BabySkyrmeEnergyTerms dmi_terms = dmi_model.energy_terms(helix);
    const double derivative_scale =
        std::sin(wave_number * dmi_lat.dx()) / dmi_lat.dx();
    const double area =
        dmi_lat.dx() * dmi_lat.dy()
        * static_cast<double>(dmi_lat.nx() * dmi_lat.ny());
    const double expected_sigma =
        0.5 * derivative_scale * derivative_scale * area;
    const double expected_dmi = dmi_strength * derivative_scale * area;

    assert(std::abs(dmi_model.dmi() - dmi_strength) < 1e-12);
    assert(std::abs(dmi_terms.sigma - expected_sigma) < 1e-10);
    assert(std::abs(dmi_terms.skyrme) < 1e-12);
    assert(std::abs(dmi_terms.potential) < 1e-12);
    assert(std::abs(dmi_terms.dmi - expected_dmi) < 1e-10);
    assert(std::abs(dmi_terms.total() - dmi_model.energy(helix)) < 1e-12);

    BabySkyrmeModel opposite_dmi_model{ 0.0, 0.0, -dmi_strength };
    const BabySkyrmeEnergyTerms opposite_dmi_terms =
        opposite_dmi_model.energy_terms(helix);

    assert(std::abs(opposite_dmi_terms.dmi + expected_dmi) < 1e-10);

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

    Lattice2D fixed_lat{
        16,
        16,
        0.5,
        0.5,
        BoundaryCondition::Fixed
    };
    O3Field fixed = SkyrmionAnsatz::charge_one(fixed_lat, 2.0);
    const Vec3 fixed_before = fixed(0, 8);
    flow.step(fixed, model);

    assert(std::abs(fixed(0, 8).x - fixed_before.x) < 1e-12);
    assert(std::abs(fixed(0, 8).y - fixed_before.y) < 1e-12);
    assert(std::abs(fixed(0, 8).z - fixed_before.z) < 1e-12);

    std::cout << "BabySkyrmeModel and BabySkyrmeGradientFlow tests passed\n";
    std::cout << "Energy before = " << energy_before << "\n";
    std::cout << "Energy after  = " << energy_after << "\n";

    return 0;
}
