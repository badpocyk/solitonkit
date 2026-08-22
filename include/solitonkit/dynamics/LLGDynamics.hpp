#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/dynamics/LandauLifshitzDynamics.hpp"
#include "solitonkit/models/MicromagneticModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

namespace solitonkit {

    class LLGDynamics {
    public:
        LLGDynamics(
            double time_step,
            double damping = 0.0,
            double gyromagnetic_ratio = 1.0
        ) : time_step_(time_step),
            damping_(damping),
            gyromagnetic_ratio_(gyromagnetic_ratio)
        {
            if (time_step_ <= 0.0 || gyromagnetic_ratio_ <= 0.0) {
                throw std::runtime_error(
                    "LLG time step and gyromagnetic ratio must be positive"
                );
            }
            if (damping_ < 0.0) {
                throw std::runtime_error("LLG damping must be non-negative");
            }
        }

        double time_step() const { return time_step_; }
        double damping() const { return damping_; }
        double gyromagnetic_ratio() const { return gyromagnetic_ratio_; }

        Vec3 direction(const Vec3& magnetization, const Vec3& field) const {
            const Vec3 m = magnetization.normalized();
            const double scale = -gyromagnetic_ratio_
                / (1.0 + damping_ * damping_);
            return scale * (
                cross(m, field)
                + damping_ * cross(m, cross(m, field))
            );
        }

        void step(O3Field& field, const MicromagneticModel& model) const {
            field.enforce_boundary_condition();
            const auto& lattice = field.lattice();
            O3Field predictor = field;

            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    if (lattice.is_fixed_boundary(i, j)) {
                        continue;
                    }
                    const Vec3 k1 = direction(
                        field(i, j),
                        model.effective_field_at(field, i, j)
                    );
                    predictor(i, j) = (field(i, j) + time_step_ * k1).normalized();
                }
            }
            predictor.enforce_boundary_condition();

            O3Field updated = field;
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    if (lattice.is_fixed_boundary(i, j)) {
                        continue;
                    }
                    const Vec3 k1 = direction(
                        field(i, j),
                        model.effective_field_at(field, i, j)
                    );
                    const Vec3 k2 = direction(
                        predictor(i, j),
                        model.effective_field_at(predictor, i, j)
                    );
                    updated(i, j) = (
                        field(i, j) + 0.5 * time_step_ * (k1 + k2)
                    ).normalized();
                }
            }
            field = updated;
            field.enforce_boundary_condition();
        }

        std::vector<DynamicsRecord> run(
            O3Field& field,
            const MicromagneticModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            if (record_every == 0) {
                throw std::runtime_error("record_every must be positive");
            }
            std::vector<DynamicsRecord> history{
                { 0, 0.0, model.energy(field), TopologicalCharge::total(field) }
            };
            for (std::size_t index = 1; index <= steps; ++index) {
                step(field, model);
                if (index % record_every == 0 || index == steps) {
                    history.push_back({
                        index,
                        static_cast<double>(index) * time_step_,
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
        double gyromagnetic_ratio_{};
    };

} // namespace solitonkit
