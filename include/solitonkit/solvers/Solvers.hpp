#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/O3Field3D.hpp"
#include "solitonkit/core/ScalarField2D.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/models/Model.hpp"

namespace solitonkit {

    struct SolverRecord {
        std::size_t step{};
        double time{};
        double energy{};
        double gradient_norm{};
        bool converged{};
    };

    struct MinimizeOptions {
        std::size_t max_steps{ 1000 };
        double step_size{ 1e-2 };
        double tolerance{ 1e-8 };
        std::size_t record_every{ 10 };
        bool line_search{ true };
        double min_step_size{ 1e-12 };
    };

    struct SolveOptions {
        std::size_t steps{ 1000 };
        double time_step{ 1e-3 };
        std::size_t record_every{ 10 };
        double tolerance{ 0.0 };
    };

    namespace detail {

        inline bool fixed_at_index(
            const ScalarField2D& field,
            std::size_t index
        ) {
            const auto& lattice = field.lattice();
            const std::size_t i = index % lattice.nx();
            const std::size_t j = index / lattice.nx();
            return lattice.is_fixed_boundary(i, j);
        }

        inline bool fixed_at_index(
            const O3Field& field,
            std::size_t index
        ) {
            const auto& lattice = field.lattice();
            const std::size_t i = index % lattice.nx();
            const std::size_t j = index / lattice.nx();
            return lattice.is_fixed_boundary(i, j);
        }

        inline bool fixed_at_index(
            const O3Field3D& field,
            std::size_t index
        ) {
            const auto& lattice = field.lattice();
            const std::size_t plane = lattice.nx() * lattice.ny();
            const std::size_t k = index / plane;
            const std::size_t remainder = index % plane;
            const std::size_t j = remainder / lattice.nx();
            const std::size_t i = remainder % lattice.nx();
            return lattice.is_fixed_boundary(i, j, k);
        }

        inline void validate(const MinimizeOptions& options) {
            if (options.step_size <= 0.0 || options.min_step_size <= 0.0) {
                throw std::runtime_error("minimizer step sizes must be positive");
            }
            if (options.tolerance < 0.0) {
                throw std::runtime_error("minimizer tolerance must be non-negative");
            }
            if (options.record_every == 0) {
                throw std::runtime_error("record_every must be positive");
            }
        }

        inline void validate(const SolveOptions& options) {
            if (options.time_step <= 0.0) {
                throw std::runtime_error("solver time step must be positive");
            }
            if (options.tolerance < 0.0) {
                throw std::runtime_error("solver tolerance must be non-negative");
            }
            if (options.record_every == 0) {
                throw std::runtime_error("record_every must be positive");
            }
        }

        template <typename Field>
        double scalar_gradient_norm(
            const Field& field,
            const std::vector<double>& gradient
        ) {
            if (gradient.size() != field.size()) {
                throw std::runtime_error("model gradient size does not match field");
            }
            double squared = 0.0;
            for (std::size_t index = 0; index < gradient.size(); ++index) {
                if (!fixed_at_index(field, index)) {
                    squared += gradient[index] * gradient[index];
                }
            }
            return std::sqrt(squared / std::max<std::size_t>(1, field.size()));
        }

        template <typename Field>
        std::vector<Vec3> tangent_gradient(
            const Field& field,
            const std::vector<Vec3>& gradient
        ) {
            if (gradient.size() != field.size()) {
                throw std::runtime_error("model gradient size does not match field");
            }
            std::vector<Vec3> result(field.size());
            for (std::size_t index = 0; index < field.size(); ++index) {
                if (fixed_at_index(field, index)) {
                    result[index] = {};
                    continue;
                }
                const Vec3 normal = field.at_index(index).normalized();
                result[index] = gradient[index]
                    - normal * dot(normal, gradient[index]);
            }
            return result;
        }

        inline double vector_gradient_norm(
            const std::vector<Vec3>& gradient
        ) {
            double squared = 0.0;
            for (const auto& value : gradient) {
                squared += value.norm_squared();
            }
            return std::sqrt(
                squared / std::max<std::size_t>(1, gradient.size())
            );
        }

        template <typename Field>
        Field scalar_step(
            const Field& field,
            const std::vector<double>& direction,
            double step_size
        ) {
            Field candidate = field;
            for (std::size_t index = 0; index < field.size(); ++index) {
                if (fixed_at_index(field, index)) {
                    continue;
                }
                candidate.at_index(index) += step_size * direction[index];
            }
            candidate.enforce_boundary_condition();
            return candidate;
        }

        template <typename Field>
        Field sphere_step(
            const Field& field,
            const std::vector<Vec3>& tangent_direction,
            double step_size
        ) {
            Field candidate = field;
            for (std::size_t index = 0; index < field.size(); ++index) {
                if (fixed_at_index(field, index)) {
                    continue;
                }
                const Vec3 normal = field.at_index(index).normalized();
                const Vec3 tangent = step_size * tangent_direction[index];
                const double angle = tangent.norm();
                if (angle < 1e-15) {
                    candidate.at_index(index) = normal;
                }
                else {
                    candidate.at_index(index) = std::cos(angle) * normal
                        + (std::sin(angle) / angle) * tangent;
                }
            }
            candidate.enforce_boundary_condition();
            return candidate;
        }

        template <typename Field>
        std::vector<SolverRecord> minimize_scalar(
            Field& field,
            const DifferentiableModel<Field, double>& model,
            const MinimizeOptions& options
        ) {
            validate(options);
            field.enforce_boundary_condition();
            double energy = model.energy(field);
            auto direction = model.negative_gradient(field);
            double norm = scalar_gradient_norm(field, direction);
            std::vector<SolverRecord> history{
                { 0, 0.0, energy, norm, norm <= options.tolerance }
            };

            if (norm <= options.tolerance) {
                return history;
            }

            for (std::size_t step = 1; step <= options.max_steps; ++step) {
                double accepted_step = options.step_size;
                Field candidate = scalar_step(field, direction, accepted_step);
                double candidate_energy = model.energy(candidate);

                if (options.line_search) {
                    while ((!std::isfinite(candidate_energy)
                            || candidate_energy > energy)
                        && accepted_step > options.min_step_size) {
                        accepted_step *= 0.5;
                        candidate = scalar_step(field, direction, accepted_step);
                        candidate_energy = model.energy(candidate);
                    }
                }

                if (!std::isfinite(candidate_energy)
                    || (options.line_search && candidate_energy > energy)) {
                    history.push_back({ step, 0.0, energy, norm, false });
                    break;
                }

                field = candidate;
                energy = candidate_energy;
                direction = model.negative_gradient(field);
                norm = scalar_gradient_norm(field, direction);
                const bool converged = norm <= options.tolerance;
                if (step % options.record_every == 0
                    || step == options.max_steps || converged) {
                    history.push_back({ step, 0.0, energy, norm, converged });
                }
                if (converged) {
                    break;
                }
            }
            return history;
        }

        template <typename Field>
        std::vector<SolverRecord> minimize_sphere(
            Field& field,
            const DifferentiableModel<Field, Vec3>& model,
            const MinimizeOptions& options
        ) {
            validate(options);
            field.enforce_boundary_condition();
            double energy = model.energy(field);
            auto direction = tangent_gradient(field, model.negative_gradient(field));
            double norm = vector_gradient_norm(direction);
            std::vector<SolverRecord> history{
                { 0, 0.0, energy, norm, norm <= options.tolerance }
            };
            if (norm <= options.tolerance) {
                return history;
            }

            for (std::size_t step = 1; step <= options.max_steps; ++step) {
                double accepted_step = options.step_size;
                Field candidate = sphere_step(field, direction, accepted_step);
                double candidate_energy = model.energy(candidate);
                if (options.line_search) {
                    while ((!std::isfinite(candidate_energy)
                            || candidate_energy > energy)
                        && accepted_step > options.min_step_size) {
                        accepted_step *= 0.5;
                        candidate = sphere_step(field, direction, accepted_step);
                        candidate_energy = model.energy(candidate);
                    }
                }
                if (!std::isfinite(candidate_energy)
                    || (options.line_search && candidate_energy > energy)) {
                    history.push_back({ step, 0.0, energy, norm, false });
                    break;
                }

                field = candidate;
                energy = candidate_energy;
                direction = tangent_gradient(field, model.negative_gradient(field));
                norm = vector_gradient_norm(direction);
                const bool converged = norm <= options.tolerance;
                if (step % options.record_every == 0
                    || step == options.max_steps || converged) {
                    history.push_back({ step, 0.0, energy, norm, converged });
                }
                if (converged) {
                    break;
                }
            }
            return history;
        }

        template <typename Field>
        std::vector<SolverRecord> solve_scalar(
            Field& field,
            const DifferentiableModel<Field, double>& model,
            const SolveOptions& options
        ) {
            validate(options);
            field.enforce_boundary_condition();
            auto direction = model.negative_gradient(field);
            double norm = scalar_gradient_norm(field, direction);
            std::vector<SolverRecord> history{
                { 0, 0.0, model.energy(field), norm,
                    options.tolerance > 0.0 && norm <= options.tolerance }
            };
            for (std::size_t step = 1; step <= options.steps; ++step) {
                field = scalar_step(field, direction, options.time_step);
                direction = model.negative_gradient(field);
                norm = scalar_gradient_norm(field, direction);
                const bool converged = options.tolerance > 0.0
                    && norm <= options.tolerance;
                if (step % options.record_every == 0
                    || step == options.steps || converged) {
                    history.push_back({
                        step,
                        static_cast<double>(step) * options.time_step,
                        model.energy(field),
                        norm,
                        converged
                    });
                }
                if (converged) {
                    break;
                }
            }
            return history;
        }

        template <typename Field>
        std::vector<SolverRecord> solve_sphere(
            Field& field,
            const DifferentiableModel<Field, Vec3>& model,
            const SolveOptions& options
        ) {
            validate(options);
            field.enforce_boundary_condition();
            auto direction = tangent_gradient(field, model.negative_gradient(field));
            double norm = vector_gradient_norm(direction);
            std::vector<SolverRecord> history{
                { 0, 0.0, model.energy(field), norm,
                    options.tolerance > 0.0 && norm <= options.tolerance }
            };
            for (std::size_t step = 1; step <= options.steps; ++step) {
                field = sphere_step(field, direction, options.time_step);
                direction = tangent_gradient(field, model.negative_gradient(field));
                norm = vector_gradient_norm(direction);
                const bool converged = options.tolerance > 0.0
                    && norm <= options.tolerance;
                if (step % options.record_every == 0
                    || step == options.steps || converged) {
                    history.push_back({
                        step,
                        static_cast<double>(step) * options.time_step,
                        model.energy(field),
                        norm,
                        converged
                    });
                }
                if (converged) {
                    break;
                }
            }
            return history;
        }

    } // namespace detail

    inline std::vector<SolverRecord> minimize(
        ScalarField2D& field,
        const DifferentiableModel<ScalarField2D, double>& model,
        const MinimizeOptions& options = {}
    ) {
        return detail::minimize_scalar(field, model, options);
    }

    inline std::vector<SolverRecord> minimize(
        XYField& field,
        const DifferentiableModel<XYField, double>& model,
        const MinimizeOptions& options = {}
    ) {
        return detail::minimize_scalar(field, model, options);
    }

    inline std::vector<SolverRecord> minimize(
        O3Field& field,
        const DifferentiableModel<O3Field, Vec3>& model,
        const MinimizeOptions& options = {}
    ) {
        return detail::minimize_sphere(field, model, options);
    }

    inline std::vector<SolverRecord> minimize(
        O3Field3D& field,
        const DifferentiableModel<O3Field3D, Vec3>& model,
        const MinimizeOptions& options = {}
    ) {
        return detail::minimize_sphere(field, model, options);
    }

    inline std::vector<SolverRecord> solve(
        ScalarField2D& field,
        const DifferentiableModel<ScalarField2D, double>& model,
        const SolveOptions& options = {}
    ) {
        return detail::solve_scalar(field, model, options);
    }

    inline std::vector<SolverRecord> solve(
        XYField& field,
        const DifferentiableModel<XYField, double>& model,
        const SolveOptions& options = {}
    ) {
        return detail::solve_scalar(field, model, options);
    }

    inline std::vector<SolverRecord> solve(
        O3Field& field,
        const DifferentiableModel<O3Field, Vec3>& model,
        const SolveOptions& options = {}
    ) {
        return detail::solve_sphere(field, model, options);
    }

    inline std::vector<SolverRecord> solve(
        O3Field3D& field,
        const DifferentiableModel<O3Field3D, Vec3>& model,
        const SolveOptions& options = {}
    ) {
        return detail::solve_sphere(field, model, options);
    }

} // namespace solitonkit
