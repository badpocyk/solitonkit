#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/flows/BabySkyrmeGradientFlow.hpp"
#include "solitonkit/flows/GradientFlow.hpp"
#include "solitonkit/models/BabySkyrmeModel.hpp"
#include "solitonkit/observables/TopologicalCharge.hpp"

namespace solitonkit {

    class BabySkyrmeOptimizerBase {
    public:
        static std::vector<Vec3> descent_field(
            const O3Field& field,
            const BabySkyrmeModel& model
        ) {
            const auto& lat = field.lattice();
            std::vector<Vec3> result(field.size());

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    const std::size_t k = lat.index(i, j);

                    if (lat.is_fixed_boundary(i, j)) {
                        result[k] = Vec3{};
                        continue;
                    }

                    result[k] = BabySkyrmeGradientFlow::projected_direction_at(
                        field,
                        model,
                        i,
                        j
                    );
                }
            }

            return result;
        }

        static std::vector<Vec3> gradient_field(
            const O3Field& field,
            const BabySkyrmeModel& model
        ) {
            std::vector<Vec3> result = descent_field(field, model);

            for (auto& value : result) {
                value *= -1.0;
            }

            return result;
        }

        static O3Field exponential_step(
            const O3Field& field,
            const std::vector<Vec3>& direction,
            double step_size
        ) {
            if (direction.size() != field.size()) {
                throw std::runtime_error("direction size does not match field size");
            }

            const auto& lat = field.lattice();
            O3Field updated = field;

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    if (lat.is_fixed_boundary(i, j)) {
                        continue;
                    }

                    const std::size_t k = lat.index(i, j);
                    const Vec3 normal = field(i, j).normalized();
                    const Vec3 tangent =
                        GradientFlow::project_to_tangent(
                            normal,
                            direction[k] * step_size
                        );
                    const double theta = tangent.norm();

                    if (theta < 1e-14) {
                        updated(i, j) = normal;
                    }
                    else {
                        updated(i, j) =
                            std::cos(theta) * normal
                            + (std::sin(theta) / theta) * tangent;
                    }
                }
            }

            updated.normalize_all();
            updated.enforce_boundary_condition();

            return updated;
        }

        static void apply_exponential_step(
            O3Field& field,
            const std::vector<Vec3>& direction,
            double step_size
        ) {
            field = exponential_step(field, direction, step_size);
        }

        static double dot_field(
            const O3Field& reference,
            const std::vector<Vec3>& a,
            const std::vector<Vec3>& b
        ) {
            if (a.size() != b.size() || a.size() != reference.size()) {
                throw std::runtime_error("vector field sizes do not match");
            }

            const auto& lat = reference.lattice();
            double result = 0.0;

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    if (lat.is_fixed_boundary(i, j)) {
                        continue;
                    }

                    const std::size_t k = lat.index(i, j);
                    result += dot(a[k], b[k]) * lat.dx() * lat.dy();
                }
            }

            return result;
        }

        static std::vector<Vec3> displacement_field(
            const O3Field& before,
            const O3Field& after
        ) {
            const auto& lat = before.lattice();
            std::vector<Vec3> result(before.size());

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    const std::size_t k = lat.index(i, j);

                    if (lat.is_fixed_boundary(i, j)) {
                        result[k] = Vec3{};
                        continue;
                    }

                    const Vec3 normal = after(i, j).normalized();
                    result[k] = GradientFlow::project_to_tangent(
                        normal,
                        after(i, j) - before(i, j)
                    );
                }
            }

            return result;
        }

        static std::vector<Vec3> difference(
            const std::vector<Vec3>& a,
            const std::vector<Vec3>& b
        ) {
            if (a.size() != b.size()) {
                throw std::runtime_error("vector field sizes do not match");
            }

            std::vector<Vec3> result(a.size());

            for (std::size_t k = 0; k < a.size(); ++k) {
                result[k] = a[k] - b[k];
            }

            return result;
        }

        static std::vector<Vec3> scaled(
            const std::vector<Vec3>& values,
            double scale
        ) {
            std::vector<Vec3> result(values.size());

            for (std::size_t k = 0; k < values.size(); ++k) {
                result[k] = values[k] * scale;
            }

            return result;
        }

        static void axpy(
            std::vector<Vec3>& y,
            double alpha,
            const std::vector<Vec3>& x
        ) {
            if (x.size() != y.size()) {
                throw std::runtime_error("vector field sizes do not match");
            }

            for (std::size_t k = 0; k < y.size(); ++k) {
                y[k] += alpha * x[k];
            }
        }

        static std::vector<Vec3> project_to_tangent_field(
            const O3Field& field,
            const std::vector<Vec3>& values
        ) {
            if (values.size() != field.size()) {
                throw std::runtime_error("vector field size does not match field size");
            }

            const auto& lat = field.lattice();
            std::vector<Vec3> result(values.size());

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    const std::size_t k = lat.index(i, j);

                    if (lat.is_fixed_boundary(i, j)) {
                        result[k] = Vec3{};
                        continue;
                    }

                    result[k] = GradientFlow::project_to_tangent(
                        field(i, j).normalized(),
                        values[k]
                    );
                }
            }

            return result;
        }

        static FlowRecord make_record(
            const O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t step
        ) {
            return {
                step,
                model.energy(field),
                TopologicalCharge::total(field)
            };
        }

        static void validate_field(const O3Field& field) {
            const auto& lat = field.lattice();

            if (lat.nx() < 3 || lat.ny() < 3) {
                throw std::runtime_error(
                    "Baby Skyrme optimization requires at least a 3x3 field"
                );
            }
        }

        static void validate_record_every(std::size_t record_every) {
            if (record_every == 0) {
                throw std::runtime_error("record_every must be positive");
            }
        }

        static void validate_step_size(double step_size, const char* name) {
            if (step_size <= 0.0) {
                throw std::runtime_error(std::string(name) + " must be positive");
            }
        }

        static double clamp(double value, double min_value, double max_value) {
            return std::max(min_value, std::min(max_value, value));
        }
    };

    class BabySkyrmeRiemannianGradientDescent
        : public BabySkyrmeOptimizerBase {
    public:
        explicit BabySkyrmeRiemannianGradientDescent(double step_size)
            : step_size_(step_size)
        {
            validate_step_size(step_size_, "step_size");
        }

        double step_size() const {
            return step_size_;
        }

        void step(O3Field& field, const BabySkyrmeModel& model) const {
            validate_field(field);
            field.enforce_boundary_condition();

            const auto direction = descent_field(field, model);
            apply_exponential_step(field, direction, step_size_);
        }

        std::vector<FlowRecord> run(
            O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            validate_field(field);
            validate_record_every(record_every);
            field.enforce_boundary_condition();

            std::vector<FlowRecord> history;
            history.push_back(make_record(field, model, 0));

            for (std::size_t s = 1; s <= steps; ++s) {
                step(field, model);

                if (s % record_every == 0 || s == steps) {
                    history.push_back(make_record(field, model, s));
                }
            }

            return history;
        }

    private:
        double step_size_{};
    };

    class BabySkyrmeBarzilaiBorweinGradient
        : public BabySkyrmeOptimizerBase {
    public:
        BabySkyrmeBarzilaiBorweinGradient(
            double initial_step_size,
            double min_step_size = 1e-8,
            double max_step_size = 1e-2,
            std::size_t max_line_search_steps = 12
        )
            : initial_step_size_(initial_step_size),
              min_step_size_(min_step_size),
              max_step_size_(max_step_size),
              max_line_search_steps_(max_line_search_steps)
        {
            validate_step_size(initial_step_size_, "initial_step_size");
            validate_step_size(min_step_size_, "min_step_size");
            validate_step_size(max_step_size_, "max_step_size");

            if (min_step_size_ > max_step_size_) {
                throw std::runtime_error("min_step_size must not exceed max_step_size");
            }

            if (max_line_search_steps_ == 0) {
                throw std::runtime_error("max_line_search_steps must be positive");
            }
        }

        std::vector<FlowRecord> run(
            O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            validate_field(field);
            validate_record_every(record_every);
            field.enforce_boundary_condition();

            std::vector<FlowRecord> history;
            history.push_back(make_record(field, model, 0));

            double step_size = clamp(
                initial_step_size_,
                min_step_size_,
                max_step_size_
            );

            auto gradient = gradient_field(field, model);

            for (std::size_t s = 1; s <= steps; ++s) {
                const O3Field before = field;
                const double energy_before = model.energy(field);
                const auto descent = scaled(gradient, -1.0);

                O3Field candidate = field;
                double accepted_step = step_size;
                bool accepted = false;

                for (std::size_t ls = 0; ls < max_line_search_steps_; ++ls) {
                    candidate = exponential_step(before, descent, accepted_step);
                    const double energy_after = model.energy(candidate);

                    if (
                        std::isfinite(energy_after)
                        && energy_after <= energy_before + 1e-12
                    ) {
                        accepted = true;
                        break;
                    }

                    accepted_step = std::max(0.5 * accepted_step, min_step_size_);
                }

                if (!accepted) {
                    candidate = before;
                    accepted_step = min_step_size_;
                }

                field = candidate;
                field.enforce_boundary_condition();

                auto new_gradient = gradient_field(field, model);
                const auto s_field = displacement_field(before, field);
                const auto y_field = difference(new_gradient, gradient);
                const double sy = dot_field(field, s_field, y_field);
                const double ss = dot_field(field, s_field, s_field);

                if (std::abs(sy) > 1e-18 && ss > 0.0) {
                    step_size = clamp(ss / std::abs(sy), min_step_size_, max_step_size_);
                }
                else {
                    step_size = accepted_step;
                }

                gradient = new_gradient;

                if (s % record_every == 0 || s == steps) {
                    history.push_back(make_record(field, model, s));
                }
            }

            return history;
        }

    private:
        double initial_step_size_{};
        double min_step_size_{};
        double max_step_size_{};
        std::size_t max_line_search_steps_{};
    };

    class BabySkyrmeLBFGSOptimizer
        : public BabySkyrmeOptimizerBase {
    public:
        BabySkyrmeLBFGSOptimizer(
            double initial_step_size = 1.0,
            std::size_t memory = 5,
            std::size_t max_line_search_steps = 12
        )
            : initial_step_size_(initial_step_size),
              memory_(memory),
              max_line_search_steps_(max_line_search_steps)
        {
            validate_step_size(initial_step_size_, "initial_step_size");

            if (memory_ == 0) {
                throw std::runtime_error("memory must be positive");
            }

            if (max_line_search_steps_ == 0) {
                throw std::runtime_error("max_line_search_steps must be positive");
            }
        }

        std::vector<FlowRecord> run(
            O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            validate_field(field);
            validate_record_every(record_every);
            field.enforce_boundary_condition();

            std::vector<FlowRecord> history;
            history.push_back(make_record(field, model, 0));

            std::deque<std::vector<Vec3>> s_history;
            std::deque<std::vector<Vec3>> y_history;
            std::deque<double> rho_history;

            auto gradient = gradient_field(field, model);

            for (std::size_t s = 1; s <= steps; ++s) {
                const O3Field before = field;
                const double energy_before = model.energy(field);
                auto direction = search_direction(
                    field,
                    gradient,
                    s_history,
                    y_history,
                    rho_history
                );

                double step_length = initial_step_size_;
                O3Field candidate = field;
                bool accepted = false;

                for (std::size_t ls = 0; ls < max_line_search_steps_; ++ls) {
                    candidate = exponential_step(before, direction, step_length);
                    const double energy_after = model.energy(candidate);

                    if (
                        std::isfinite(energy_after)
                        && energy_after <= energy_before + 1e-12
                    ) {
                        accepted = true;
                        break;
                    }

                    step_length *= 0.5;
                }

                if (!accepted) {
                    direction = scaled(gradient, -1.0);
                    step_length = initial_step_size_;

                    for (std::size_t ls = 0; ls < max_line_search_steps_; ++ls) {
                        candidate = exponential_step(before, direction, step_length);
                        const double energy_after = model.energy(candidate);

                        if (
                            std::isfinite(energy_after)
                            && energy_after <= energy_before + 1e-12
                        ) {
                            accepted = true;
                            break;
                        }

                        step_length *= 0.5;
                    }
                }

                if (!accepted) {
                    candidate = before;
                }

                field = candidate;
                field.enforce_boundary_condition();

                auto new_gradient = gradient_field(field, model);
                const auto s_field = displacement_field(before, field);
                const auto y_field = difference(new_gradient, gradient);
                const double sy = dot_field(field, s_field, y_field);

                if (sy > 1e-18) {
                    if (s_history.size() == memory_) {
                        s_history.pop_front();
                        y_history.pop_front();
                        rho_history.pop_front();
                    }

                    s_history.push_back(s_field);
                    y_history.push_back(y_field);
                    rho_history.push_back(1.0 / sy);
                }

                gradient = new_gradient;

                if (s % record_every == 0 || s == steps) {
                    history.push_back(make_record(field, model, s));
                }
            }

            return history;
        }

    private:
        double initial_step_size_{};
        std::size_t memory_{};
        std::size_t max_line_search_steps_{};

        std::vector<Vec3> search_direction(
            const O3Field& field,
            const std::vector<Vec3>& gradient,
            const std::deque<std::vector<Vec3>>& s_history,
            const std::deque<std::vector<Vec3>>& y_history,
            const std::deque<double>& rho_history
        ) const {
            if (s_history.empty()) {
                return scaled(gradient, -1.0);
            }

            std::vector<Vec3> q = gradient;
            std::vector<double> alpha(s_history.size());

            for (std::size_t offset = 0; offset < s_history.size(); ++offset) {
                const std::size_t index = s_history.size() - 1 - offset;
                alpha[index] =
                    rho_history[index] * dot_field(field, s_history[index], q);
                axpy(q, -alpha[index], y_history[index]);
            }

            const auto& last_s = s_history.back();
            const auto& last_y = y_history.back();
            const double yy = dot_field(field, last_y, last_y);
            const double sy = dot_field(field, last_s, last_y);
            const double gamma =
                (yy > 1e-18 && sy > 0.0) ? sy / yy : 1.0;

            std::vector<Vec3> r = scaled(q, gamma);

            for (std::size_t index = 0; index < s_history.size(); ++index) {
                const double beta =
                    rho_history[index] * dot_field(field, y_history[index], r);
                axpy(r, alpha[index] - beta, s_history[index]);
            }

            return project_to_tangent_field(field, scaled(r, -1.0));
        }
    };

    class BabySkyrmeSemiImplicitFlow
        : public BabySkyrmeOptimizerBase {
    public:
        BabySkyrmeSemiImplicitFlow(
            double step_size,
            std::size_t implicit_iterations = 20
        )
            : step_size_(step_size),
              implicit_iterations_(implicit_iterations)
        {
            validate_step_size(step_size_, "step_size");

            if (implicit_iterations_ == 0) {
                throw std::runtime_error("implicit_iterations must be positive");
            }
        }

        void step(O3Field& field, const BabySkyrmeModel& model) const {
            validate_field(field);
            field.enforce_boundary_condition();

            const auto& lat = field.lattice();
            O3Field rhs = field;

            for (std::size_t j = 0; j < lat.ny(); ++j) {
                for (std::size_t i = 0; i < lat.nx(); ++i) {
                    if (lat.is_fixed_boundary(i, j)) {
                        continue;
                    }

                    const Vec3 normal = field(i, j).normalized();
                    const Vec3 explicit_force =
                        BabySkyrmeGradientFlow::skyrme_force_at(field, model, i, j)
                        + BabySkyrmeGradientFlow::dmi_force_at(field, model, i, j)
                        + Vec3{ 0.0, 0.0, model.mass() * model.mass() };
                    const Vec3 tangent_force =
                        GradientFlow::project_to_tangent(normal, explicit_force);

                    rhs(i, j) = normal + step_size_ * tangent_force;
                }
            }

            O3Field current = field;
            const double cx = step_size_ / (lat.dx() * lat.dx());
            const double cy = step_size_ / (lat.dy() * lat.dy());
            const double denominator = 1.0 + 2.0 * cx + 2.0 * cy;

            for (std::size_t iteration = 0; iteration < implicit_iterations_; ++iteration) {
                O3Field next = current;

                for (std::size_t j = 0; j < lat.ny(); ++j) {
                    for (std::size_t i = 0; i < lat.nx(); ++i) {
                        if (lat.is_fixed_boundary(i, j)) {
                            continue;
                        }

                        const Vec3 value =
                            (
                                rhs(i, j)
                                + cx * (
                                    current(lat.left(i), j)
                                    + current(lat.right(i), j)
                                )
                                + cy * (
                                    current(i, lat.down(j))
                                    + current(i, lat.up(j))
                                )
                            ) / denominator;

                        next(i, j) = value.normalized();
                    }
                }

                next.enforce_boundary_condition();
                current = next;
            }

            field = current;
            field.enforce_boundary_condition();
        }

        std::vector<FlowRecord> run(
            O3Field& field,
            const BabySkyrmeModel& model,
            std::size_t steps,
            std::size_t record_every = 1
        ) const {
            validate_field(field);
            validate_record_every(record_every);
            field.enforce_boundary_condition();

            std::vector<FlowRecord> history;
            history.push_back(make_record(field, model, 0));

            for (std::size_t s = 1; s <= steps; ++s) {
                step(field, model);

                if (s % record_every == 0 || s == steps) {
                    history.push_back(make_record(field, model, s));
                }
            }

            return history;
        }

    private:
        double step_size_{};
        std::size_t implicit_iterations_{};
    };

} // namespace solitonkit
