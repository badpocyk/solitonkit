#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/flows/BabySkyrmeOptimizers.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"

using namespace solitonkit;

namespace {

    O3Field make_perturbed_field() {
        Lattice2D lat{
            16,
            16,
            0.5,
            0.5,
            BoundaryCondition::Dirichlet
        };

        O3Field field = O3Field::uniform(lat, Vec3{ 0.0, 0.0, 1.0 });
        field(8, 8) = Vec3{ 1.0, 0.0, 0.0 };
        field(7, 8) = Vec3{ 0.8, 0.2, 0.5 }.normalized();
        field.normalize_all();

        return field;
    }

    void assert_unit_norms_and_vacuum_boundary(const O3Field& field) {
        const auto& lat = field.lattice();

        for (std::size_t j = 0; j < lat.ny(); ++j) {
            for (std::size_t i = 0; i < lat.nx(); ++i) {
                const Vec3 value = field(i, j);

                assert(std::abs(value.norm() - 1.0) < 1e-12);

                if (lat.is_boundary(i, j)) {
                    assert(std::abs(value.x) < 1e-12);
                    assert(std::abs(value.y) < 1e-12);
                    assert(std::abs(value.z - 1.0) < 1e-12);
                }
            }
        }
    }

    template <typename Optimizer>
    void assert_optimizer_relaxes(const Optimizer& optimizer, const char* name) {
        BabySkyrmeModel model{ 0.0, 0.0, 0.0 };
        O3Field field = make_perturbed_field();
        const double before = model.energy(field);

        const auto records = optimizer.run(field, model, 4, 1);
        const double after = model.energy(field);

        assert(records.size() == 5);
        assert(std::isfinite(before));
        assert(std::isfinite(after));
        assert(after <= before + 1e-10);
        assert_unit_norms_and_vacuum_boundary(field);

        std::cout << name << " energy before = " << before << "\n";
        std::cout << name << " energy after  = " << after << "\n";
    }

} // namespace

int main() {
    assert_optimizer_relaxes(
        BabySkyrmeRiemannianGradientDescent{ 1e-2 },
        "RiemannianGradientDescent"
    );
    assert_optimizer_relaxes(
        BabySkyrmeBarzilaiBorweinGradient{ 1e-2, 1e-8, 1e-1 },
        "BarzilaiBorweinGradient"
    );
    assert_optimizer_relaxes(
        BabySkyrmeLBFGSOptimizer{ 1.0, 4, 12 },
        "LBFGS"
    );
    assert_optimizer_relaxes(
        BabySkyrmeSemiImplicitFlow{ 1e-2, 10 },
        "SemiImplicitFlow"
    );

    std::cout << "Baby Skyrme optimizer tests passed\n";

    return 0;
}
