#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/models/O3SigmaModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

namespace solitonkit {

    struct FlowRecord {
        std::size_t step{};
        double energy{};
        double topological_charge{};
    };

    class GradientFlow {
    public:
        GradientFlow()
            : step_size_(0.01)
        {}

        explicit GradientFlow(double step_size)
            : step_size_(step_size)
        {
            validate_step_size(step_size_);
        }

        std::string description() const {
            return "GradientFlow(projected O(3) sigma-model flow)";
        }

        double step_size() const {
            return step_size_;
        }

        void set_step_size(double step_size) {
            validate_step_size(step_size);
            step_size_ = step_size;
        }

        static Vec3 laplacian_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const std::size_t il = lat.left(i);
            const std::size_t ir = lat.right(i);
            const std::size_t jd = lat.down(j);
            const std::size_t ju = lat.up(j);

            const Vec3 phi = field(i, j);

            const Vec3 lap_x =
                (field(ir, j) - 2.0 * phi + field(il, j))
                / (lat.dx() * lat.dx());

            const Vec3 lap_y =
                (field(i, ju) - 2.0 * phi + field(i, jd))
                / (lat.dy() * lat.dy());

            return lap_x + lap_y;
        }

        static Vec3 project_to_tangent(
            const Vec3& normal,
            const Vec3& vector
        ) {
            return vector - normal * dot(normal, vector);
        }

        static Vec3 projected_laplacian_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const Vec3 phi = field(i, j);
            const Vec3 lap = laplacian_at(field, i, j);

            if (phi.norm_squared() == 0.0) {
                return lap;
            }

            const Vec3 normal = phi.normalized();

            return project_to_tangent(normal, lap);
        }

        void step(O3Field& field) const {
            field.enforce_boundary_condition();

            const auto& lat = field.lattice();

            O3Field updated = field;

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    if (lat.is_fixed_boundary(i, j)) {
                        continue;
                    }

                    const Vec3 phi = field(i, j);
                    const Vec3 direction = projected_laplacian_at(field, i, j);

                    const Vec3 new_value = phi + step_size_ * direction;

                    if (new_value.norm_squared() == 0.0) {
                        updated(i, j) = phi;
                    }
                    else {
                        updated(i, j) = new_value.normalized();
                    }
                }
            }

            field = updated;
            field.enforce_boundary_condition();
        }

        std::vector<FlowRecord> run(
            O3Field& field,
            const O3SigmaModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            if (record_every == 0) {
                throw std::runtime_error("record_every must be positive");
            }

            field.enforce_boundary_condition();

            std::vector<FlowRecord> history;

            history.push_back({
                0,
                model.total_energy(field),
                TopologicalCharge::total(field)
                });

            for (std::size_t s = 1; s <= steps; ++s) {
                step(field);

                if (s % record_every == 0 || s == steps) {
                    history.push_back({
                        s,
                        model.total_energy(field),
                        TopologicalCharge::total(field)
                        });
                }
            }

            return history;
        }

    private:
        double step_size_{};

        static void validate_step_size(double step_size) {
            if (step_size <= 0.0) {
                throw std::runtime_error("GradientFlow step size must be positive");
            }
        }
    };

} // namespace solitonkit
