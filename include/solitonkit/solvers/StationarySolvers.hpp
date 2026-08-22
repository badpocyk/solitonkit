#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "solitonkit/analysis/LinearStability.hpp"
#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/O3Field3D.hpp"
#include "solitonkit/core/ScalarField2D.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/models/Model.hpp"

namespace solitonkit {

    struct GMRESOptions {
        std::size_t restart{ 30 };
        std::size_t max_iterations{ 200 };
        double tolerance{ 1e-5 };
    };

    struct LinearSolveResult {
        std::vector<double> solution;
        double residual_norm{};
        std::size_t iterations{};
        bool converged{};
    };

    struct StationaryOptions {
        std::size_t max_steps{ 40 };
        double tolerance{ 1e-8 };
        double finite_difference_step{ 1e-5 };
        double initial_damping{ 1.0 };
        double minimum_damping{ 1e-6 };
        double trust_radius{ 10.0 };
        bool line_search{ true };
        std::size_t preconditioner_probes{ 4 };
        double preconditioner_floor{ 1e-6 };
        unsigned int seed{ 12345 };
        GMRESOptions gmres{};
    };

    struct StationaryRecord {
        std::size_t step{};
        double energy{};
        double residual_norm{};
        double damping{};
        std::size_t linear_iterations{};
        double linear_residual{};
        bool linear_converged{};
        bool converged{};
    };

    namespace stationary_detail {

        using CoordinateVector = stability_detail::CoordinateVector;
        using LinearOperator = stability_detail::LinearOperator;

        inline CoordinateVector apply_preconditioner(
            const CoordinateVector& value,
            const CoordinateVector& inverse_diagonal
        ) {
            if (value.size() != inverse_diagonal.size()) {
                throw std::runtime_error("preconditioner size mismatch");
            }
            CoordinateVector result(value.size());
            for (std::size_t index = 0; index < value.size(); ++index) {
                result[index] = inverse_diagonal[index] * value[index];
            }
            return result;
        }

        inline void validate(const GMRESOptions& options) {
            if (options.restart == 0 || options.max_iterations == 0) {
                throw std::runtime_error(
                    "GMRES restart and max_iterations must be positive"
                );
            }
            if (options.tolerance <= 0.0) {
                throw std::runtime_error("GMRES tolerance must be positive");
            }
        }

        inline void validate(const StationaryOptions& options) {
            if (options.max_steps == 0 || options.tolerance <= 0.0
                || options.finite_difference_step <= 0.0
                || options.initial_damping <= 0.0
                || options.minimum_damping <= 0.0
                || options.minimum_damping > options.initial_damping
                || options.trust_radius < 0.0
                || options.preconditioner_floor <= 0.0) {
                throw std::runtime_error("invalid stationary solver options");
            }
            validate(options.gmres);
        }

        inline CoordinateVector residual(
            const LinearOperator& apply,
            const CoordinateVector& right_hand_side,
            const CoordinateVector& solution
        ) {
            CoordinateVector result = right_hand_side;
            stability_detail::axpy(result, -1.0, apply(solution));
            return result;
        }

        inline LinearSolveResult gmres_impl(
            const LinearOperator& apply,
            const CoordinateVector& right_hand_side,
            const GMRESOptions& options,
            const CoordinateVector& inverse_diagonal
        ) {
            validate(options);
            const std::size_t size = right_hand_side.size();
            if (size == 0 || inverse_diagonal.size() != size) {
                throw std::runtime_error("invalid GMRES system size");
            }

            CoordinateVector solution(size, 0.0);
            const double right_norm = stability_detail::vector_norm(
                right_hand_side
            );
            const double target = options.tolerance
                * std::max(1.0, right_norm);
            std::size_t total_iterations = 0;

            while (total_iterations < options.max_iterations) {
                CoordinateVector current_residual = residual(
                    apply, right_hand_side, solution
                );
                const double beta = stability_detail::vector_norm(
                    current_residual
                );
                if (beta <= target) {
                    return {
                        solution, beta, total_iterations, true
                    };
                }

                const std::size_t cycle_size = std::min(
                    options.restart,
                    options.max_iterations - total_iterations
                );
                std::vector<CoordinateVector> basis;
                basis.reserve(cycle_size + 1);
                for (double& value : current_residual) {
                    value /= beta;
                }
                basis.push_back(std::move(current_residual));

                std::vector<double> hessenberg(
                    (cycle_size + 1) * cycle_size,
                    0.0
                );
                std::vector<double> cosines(cycle_size, 0.0);
                std::vector<double> sines(cycle_size, 0.0);
                std::vector<double> projected_rhs(cycle_size + 1, 0.0);
                projected_rhs[0] = beta;
                std::size_t used = 0;

                for (std::size_t column = 0;
                    column < cycle_size; ++column) {
                    CoordinateVector preconditioned = apply_preconditioner(
                        basis[column], inverse_diagonal
                    );
                    CoordinateVector work = apply(preconditioned);
                    for (int pass = 0; pass < 2; ++pass) {
                        for (std::size_t row = 0;
                            row <= column; ++row) {
                            const double projection = stability_detail::vector_dot(
                                work, basis[row]
                            );
                            hessenberg[row * cycle_size + column] += projection;
                            stability_detail::axpy(
                                work, -projection, basis[row]
                            );
                        }
                    }
                    const double next_norm = stability_detail::vector_norm(work);
                    hessenberg[(column + 1) * cycle_size + column] = next_norm;
                    if (next_norm > 1e-14) {
                        for (double& value : work) {
                            value /= next_norm;
                        }
                        basis.push_back(std::move(work));
                    }
                    else {
                        basis.push_back(CoordinateVector(size, 0.0));
                    }

                    for (std::size_t rotation = 0;
                        rotation < column; ++rotation) {
                        const double upper = hessenberg[
                            rotation * cycle_size + column
                        ];
                        const double lower = hessenberg[
                            (rotation + 1) * cycle_size + column
                        ];
                        hessenberg[rotation * cycle_size + column] =
                            cosines[rotation] * upper
                            + sines[rotation] * lower;
                        hessenberg[(rotation + 1) * cycle_size + column] =
                            -sines[rotation] * upper
                            + cosines[rotation] * lower;
                    }

                    const double diagonal = hessenberg[
                        column * cycle_size + column
                    ];
                    const double below = hessenberg[
                        (column + 1) * cycle_size + column
                    ];
                    const double norm = std::hypot(diagonal, below);
                    cosines[column] = norm <= 1e-30 ? 1.0 : diagonal / norm;
                    sines[column] = norm <= 1e-30 ? 0.0 : below / norm;
                    hessenberg[column * cycle_size + column] = norm;
                    hessenberg[(column + 1) * cycle_size + column] = 0.0;

                    const double rhs_value = projected_rhs[column];
                    const double rhs_below = projected_rhs[column + 1];
                    projected_rhs[column] = cosines[column] * rhs_value
                        + sines[column] * rhs_below;
                    projected_rhs[column + 1] = -sines[column] * rhs_value
                        + cosines[column] * rhs_below;

                    used = column + 1;
                    ++total_iterations;
                    if (std::abs(projected_rhs[column + 1]) <= target
                        || next_norm <= 1e-14
                        || total_iterations == options.max_iterations) {
                        break;
                    }
                }

                CoordinateVector coefficients(used, 0.0);
                for (std::size_t reverse = 0; reverse < used; ++reverse) {
                    const std::size_t row = used - 1 - reverse;
                    double value = projected_rhs[row];
                    for (std::size_t column = row + 1;
                        column < used; ++column) {
                        value -= hessenberg[row * cycle_size + column]
                            * coefficients[column];
                    }
                    const double diagonal = hessenberg[
                        row * cycle_size + row
                    ];
                    if (std::abs(diagonal) <= 1e-30) {
                        coefficients[row] = 0.0;
                    }
                    else {
                        coefficients[row] = value / diagonal;
                    }
                }

                CoordinateVector correction(size, 0.0);
                for (std::size_t column = 0; column < used; ++column) {
                    stability_detail::axpy(
                        correction, coefficients[column], basis[column]
                    );
                }
                correction = apply_preconditioner(
                    correction, inverse_diagonal
                );
                stability_detail::axpy(solution, 1.0, correction);

                const double actual_residual = stability_detail::vector_norm(
                    residual(apply, right_hand_side, solution)
                );
                if (actual_residual <= target) {
                    return {
                        solution,
                        actual_residual,
                        total_iterations,
                        true
                    };
                }
            }

            const double final_residual = stability_detail::vector_norm(
                residual(apply, right_hand_side, solution)
            );
            return {
                solution,
                final_residual,
                total_iterations,
                final_residual <= target
            };
        }

        inline CoordinateVector approximate_inverse_diagonal(
            std::size_t size,
            const LinearOperator& apply,
            std::size_t probes,
            double floor,
            unsigned int seed
        ) {
            CoordinateVector inverse(size, 1.0);
            if (probes == 0) {
                return inverse;
            }

            CoordinateVector diagonal(size, 0.0);
            std::mt19937 generator(seed);
            std::uniform_int_distribution<int> sign(0, 1);
            for (std::size_t probe = 0; probe < probes; ++probe) {
                CoordinateVector direction(size);
                for (double& value : direction) {
                    value = sign(generator) == 0 ? -1.0 : 1.0;
                }
                const CoordinateVector applied = apply(direction);
                for (std::size_t index = 0; index < size; ++index) {
                    diagonal[index] += direction[index] * applied[index];
                }
            }
            for (std::size_t index = 0; index < size; ++index) {
                const double estimate = std::abs(
                    diagonal[index] / static_cast<double>(probes)
                );
                inverse[index] = 1.0 / std::max(floor, estimate);
            }
            return inverse;
        }

        template <typename Field, typename Model, typename CoordinateSpace>
        std::vector<StationaryRecord> solve_impl(
            Field& field,
            const Model& model,
            const StationaryOptions& options
        ) {
            validate(options);
            field.enforce_boundary_condition();
            std::vector<StationaryRecord> history;

            for (std::size_t step = 0; step <= options.max_steps; ++step) {
                const CoordinateSpace space(
                    field, model, options.finite_difference_step
                );
                const CoordinateVector current_residual = space.residual();
                const double residual_norm = stability_detail::vector_norm(
                    current_residual
                ) / std::sqrt(static_cast<double>(space.size()));
                const bool converged = residual_norm <= options.tolerance;
                if (step == 0 || converged) {
                    history.push_back({
                        step,
                        model.energy(field),
                        residual_norm,
                        0.0,
                        0,
                        0.0,
                        true,
                        converged
                    });
                }
                if (converged || step == options.max_steps) {
                    break;
                }

                const LinearOperator hessian = [&](const CoordinateVector& value) {
                    return space.apply(value);
                };
                const CoordinateVector inverse_diagonal =
                    approximate_inverse_diagonal(
                        space.size(),
                        hessian,
                        options.preconditioner_probes,
                        options.preconditioner_floor,
                        options.seed + static_cast<unsigned int>(step)
                    );
                const LinearSolveResult linear = gmres_impl(
                    hessian,
                    current_residual,
                    options.gmres,
                    inverse_diagonal
                );

                CoordinateVector direction = linear.solution;
                const double direction_norm = stability_detail::vector_norm(
                    direction
                );
                if (options.trust_radius > 0.0
                    && direction_norm > options.trust_radius) {
                    const double scale = options.trust_radius / direction_norm;
                    for (double& value : direction) {
                        value *= scale;
                    }
                }

                double damping = options.initial_damping;
                Field candidate = space.retract(direction, damping);
                double candidate_norm = stability_detail::vector_norm(
                    space.residual_at(candidate, model)
                ) / std::sqrt(static_cast<double>(space.size()));
                if (options.line_search) {
                    while ((!std::isfinite(candidate_norm)
                            || candidate_norm >= residual_norm)
                        && damping > options.minimum_damping) {
                        damping *= 0.5;
                        candidate = space.retract(direction, damping);
                        candidate_norm = stability_detail::vector_norm(
                            space.residual_at(candidate, model)
                        ) / std::sqrt(static_cast<double>(space.size()));
                    }
                }

                const bool accepted = std::isfinite(candidate_norm)
                    && (!options.line_search || candidate_norm < residual_norm);
                history.push_back({
                    step + 1,
                    accepted ? model.energy(candidate) : model.energy(field),
                    accepted ? candidate_norm : residual_norm,
                    accepted ? damping : 0.0,
                    linear.iterations,
                    linear.residual_norm,
                    linear.converged,
                    accepted && candidate_norm <= options.tolerance
                });
                if (!accepted) {
                    break;
                }
                field = std::move(candidate);
                if (candidate_norm <= options.tolerance) {
                    break;
                }
            }
            return history;
        }

    } // namespace stationary_detail

    inline LinearSolveResult gmres(
        const stationary_detail::LinearOperator& apply,
        const std::vector<double>& right_hand_side,
        const GMRESOptions& options = {},
        const std::vector<double>& inverse_diagonal = {}
    ) {
        const std::vector<double> preconditioner = inverse_diagonal.empty()
            ? std::vector<double>(right_hand_side.size(), 1.0)
            : inverse_diagonal;
        return stationary_detail::gmres_impl(
            apply, right_hand_side, options, preconditioner
        );
    }

    inline std::vector<StationaryRecord> solve_stationary(
        ScalarField2D& field,
        const DifferentiableModel<ScalarField2D, double>& model,
        const StationaryOptions& options = {}
    ) {
        return stationary_detail::solve_impl<
            ScalarField2D,
            DifferentiableModel<ScalarField2D, double>,
            stability_detail::ScalarCoordinateSpace<ScalarField2D>
        >(field, model, options);
    }

    inline std::vector<StationaryRecord> solve_stationary(
        XYField& field,
        const DifferentiableModel<XYField, double>& model,
        const StationaryOptions& options = {}
    ) {
        return stationary_detail::solve_impl<
            XYField,
            DifferentiableModel<XYField, double>,
            stability_detail::ScalarCoordinateSpace<XYField>
        >(field, model, options);
    }

    inline std::vector<StationaryRecord> solve_stationary(
        O3Field& field,
        const DifferentiableModel<O3Field, Vec3>& model,
        const StationaryOptions& options = {}
    ) {
        return stationary_detail::solve_impl<
            O3Field,
            DifferentiableModel<O3Field, Vec3>,
            stability_detail::SphereCoordinateSpace<O3Field>
        >(field, model, options);
    }

    inline std::vector<StationaryRecord> solve_stationary(
        O3Field3D& field,
        const DifferentiableModel<O3Field3D, Vec3>& model,
        const StationaryOptions& options = {}
    ) {
        return stationary_detail::solve_impl<
            O3Field3D,
            DifferentiableModel<O3Field3D, Vec3>,
            stability_detail::SphereCoordinateSpace<O3Field3D>
        >(field, model, options);
    }

} // namespace solitonkit
