#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/flows/GradientFlow.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

namespace solitonkit {

    class BabySkyrmeGradientFlow {
    public:
        explicit BabySkyrmeGradientFlow(double step_size)
            : step_size_(step_size)
        {
            if (step_size_ <= 0.0) {
                throw std::runtime_error(
                    "BabySkyrmeGradientFlow step size must be positive"
                );
            }
        }

        double step_size() const {
            return step_size_;
        }

        static Vec3 sigma_force_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const Vec3 ddx =
                (
                    BabySkyrmeModel::derivative_x(field, lat.right(i), j)
                    - BabySkyrmeModel::derivative_x(field, lat.left(i), j)
                ) / (2.0 * lat.dx());

            const Vec3 ddy =
                (
                    BabySkyrmeModel::derivative_y(field, i, lat.up(j))
                    - BabySkyrmeModel::derivative_y(field, i, lat.down(j))
                ) / (2.0 * lat.dy());

            return ddx + ddy;
        }

        static Vec3 skyrme_flux_x_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) {
            const Vec3 dx = BabySkyrmeModel::derivative_x(field, i, j);
            const Vec3 dy = BabySkyrmeModel::derivative_y(field, i, j);
            const Vec3 area = cross(dx, dy);

            return model.kappa() * cross(dy, area);
        }

        static Vec3 skyrme_flux_y_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) {
            const Vec3 dx = BabySkyrmeModel::derivative_x(field, i, j);
            const Vec3 dy = BabySkyrmeModel::derivative_y(field, i, j);
            const Vec3 area = cross(dx, dy);

            return model.kappa() * cross(area, dx);
        }

        static Vec3 skyrme_force_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const Vec3 dx_flux =
                (
                    skyrme_flux_x_at(field, model, lat.right(i), j)
                    - skyrme_flux_x_at(field, model, lat.left(i), j)
                ) / (2.0 * lat.dx());

            const Vec3 dy_flux =
                (
                    skyrme_flux_y_at(field, model, i, lat.up(j))
                    - skyrme_flux_y_at(field, model, i, lat.down(j))
                ) / (2.0 * lat.dy());

            return dx_flux + dy_flux;
        }

        static Vec3 dmi_force_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) {
            return -2.0 * model.dmi() * BabySkyrmeModel::curl(field, i, j);
        }

        static Vec3 negative_gradient_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) {
            const Vec3 potential_force{ 0.0, 0.0, model.mass() * model.mass() };

            return sigma_force_at(field, i, j)
                + skyrme_force_at(field, model, i, j)
                + dmi_force_at(field, model, i, j)
                + potential_force;
        }

        static Vec3 projected_direction_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) {
            const Vec3 normal = field(i, j).normalized();
            const Vec3 force = negative_gradient_at(field, model, i, j);

            return GradientFlow::project_to_tangent(normal, force);
        }

        void step(O3Field& field, const BabySkyrmeModel& model) const {
            validate_field(field);

            const auto& lat = field.lattice();
            O3Field updated = field;

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    if (lat.is_fixed_boundary(i, j)) {
                        continue;
                    }

                    const Vec3 next = field(i, j)
                        + step_size_ * projected_direction_at(field, model, i, j);

                    updated(i, j) = next.normalized();
                }
            }

            field = updated;
        }

        std::vector<FlowRecord> run(
            O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            validate_field(field);

            if (record_every == 0) {
                throw std::runtime_error("record_every must be positive");
            }

            std::vector<FlowRecord> history;
            history.push_back({
                0,
                model.energy(field),
                TopologicalCharge::total(field)
            });

            for (std::size_t s = 1; s <= steps; ++s) {
                step(field, model);

                if (s % record_every == 0 || s == steps) {
                    history.push_back({
                        s,
                        model.energy(field),
                        TopologicalCharge::total(field)
                    });
                }
            }

            return history;
        }

    private:
        double step_size_{};

        static void validate_field(const O3Field& field) {
            const auto& lat = field.lattice();

            if (lat.nx() < 3 || lat.ny() < 3) {
                throw std::runtime_error(
                    "BabySkyrmeGradientFlow requires at least a 3x3 field"
                );
            }
        }
    };

} // namespace solitonkit
