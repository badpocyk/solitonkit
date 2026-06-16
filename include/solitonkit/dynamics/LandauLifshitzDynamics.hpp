#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/flows/BabySkyrmeGradientFlow.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

namespace solitonkit {

    struct DynamicsRecord {
        std::size_t step{};
        double time{};
        double energy{};
        double topological_charge{};
    };

    class LandauLifshitzDynamics {
    public:
        LandauLifshitzDynamics(double time_step, double damping = 0.0)
            : time_step_(time_step), damping_(damping)
        {
            if (time_step_ <= 0.0) {
                throw std::runtime_error(
                    "LandauLifshitzDynamics time step must be positive"
                );
            }

            if (damping_ < 0.0) {
                throw std::runtime_error(
                    "LandauLifshitzDynamics damping must be non-negative"
                );
            }
        }

        double time_step() const {
            return time_step_;
        }

        double damping() const {
            return damping_;
        }

        static Vec3 effective_field_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) {
            return BabySkyrmeGradientFlow::negative_gradient_at(
                field,
                model,
                i,
                j
            );
        }

        Vec3 direction_at(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t i,
            std::size_t j
        ) const {
            const Vec3 normal = field(i, j).normalized();
            const Vec3 effective_field = effective_field_at(field, model, i, j);

            // H = -delta E / delta n; damping follows its tangent projection.
            const Vec3 precession = -1.0 * cross(normal, effective_field);
            const Vec3 damping =
                -damping_ * cross(normal, cross(normal, effective_field));

            return precession + damping;
        }

        void step(O3Field& field, const BabySkyrmeModel& model) const {
            field.enforce_boundary_condition();

            const auto& lat = field.lattice();

            if (lat.nx() < 3 || lat.ny() < 3) {
                throw std::runtime_error(
                    "LandauLifshitzDynamics requires at least a 3x3 field"
                );
            }

            O3Field updated = field;

            // Forward Euler step followed by projection back onto S^2.
            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    if (lat.is_fixed_boundary(i, j)) {
                        continue;
                    }

                    const Vec3 next = field(i, j)
                        + time_step_ * direction_at(field, model, i, j);

                    updated(i, j) = next.normalized();
                }
            }

            field = updated;
            field.enforce_boundary_condition();
        }

        std::vector<DynamicsRecord> run(
            O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            if (record_every == 0) {
                throw std::runtime_error("record_every must be positive");
            }

            field.enforce_boundary_condition();

            std::vector<DynamicsRecord> history;
            history.push_back({
                0,
                0.0,
                model.energy(field),
                TopologicalCharge::total(field)
            });

            for (std::size_t s = 1; s <= steps; ++s) {
                step(field, model);

                if (s % record_every == 0 || s == steps) {
                    history.push_back({
                        s,
                        static_cast<double>(s) * time_step_,
                        model.energy(field),
                        TopologicalCharge::total(field)
                    });
                }
            }

            return history;
        }

    private:
        double time_step_{};
        double damping_{};
    };

} // namespace solitonkit
