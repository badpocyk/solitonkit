#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "solitonkit/analysis/LinearStability.hpp"
#include "solitonkit/solvers/StationarySolvers.hpp"

namespace solitonkit {

    struct ContinuationOptions {
        double start{};
        double stop{ 1.0 };
        double step{ 0.1 };
        double minimum_step{ 1e-4 };
        double maximum_step{ 0.25 };
        double step_growth{ 1.25 };
        double step_shrink{ 0.5 };
        double parameter_scale{ 1.0 };
        std::size_t max_points{ 100 };
        std::size_t max_corrector_steps{ 12 };
        std::size_t target_corrector_steps{ 4 };
        double corrector_tolerance{ 1e-7 };
        double finite_difference_step{ 1e-5 };
        double minimum_damping{ 1e-6 };
        double trust_radius{ 10.0 };
        std::size_t preconditioner_probes{ 2 };
        double preconditioner_floor{ 1e-6 };
        bool analyze_stability{ true };
        double bifurcation_tolerance{ 1e-5 };
        unsigned int seed{ 12345 };
        GMRESOptions gmres{};
        StationaryOptions stationary{};
        StabilityOptions stability{};
    };

    template <typename Field>
    struct ContinuationPoint {
        explicit ContinuationPoint(const Field& field_value)
            : field(field_value) {}

        double parameter{};
        double energy{};
        double residual_norm{};
        double lowest_eigenvalue{ std::numeric_limits<double>::quiet_NaN() };
        std::size_t corrector_steps{};
        bool converged{};
        bool stable{};
        bool bifurcation_candidate{};
        Field field;
    };

    template <typename Field>
    struct BranchResult {
        std::vector<ContinuationPoint<Field>> points;
        bool reached_stop{};
        bool converged{};
    };

    namespace continuation_detail {

        using CoordinateVector = stability_detail::CoordinateVector;
        using LinearOperator = stability_detail::LinearOperator;

        inline void validate(const ContinuationOptions& options) {
            if (options.step == 0.0 || options.minimum_step <= 0.0
                || options.maximum_step < options.minimum_step
                || options.step_growth < 1.0
                || options.step_shrink <= 0.0
                || options.step_shrink >= 1.0
                || options.parameter_scale <= 0.0
                || options.max_points < 2
                || options.max_corrector_steps == 0
                || options.corrector_tolerance <= 0.0
                || options.finite_difference_step <= 0.0
                || options.minimum_damping <= 0.0
                || options.trust_radius < 0.0
                || options.preconditioner_floor <= 0.0
                || options.bifurcation_tolerance < 0.0) {
                throw std::runtime_error("invalid continuation options");
            }
            stationary_detail::validate(options.gmres);
        }

        inline bool reached(
            double parameter,
            double stop,
            double direction
        ) {
            return direction * (parameter - stop) >= -1e-12;
        }

        template <typename Field, typename Model, typename CoordinateSpace>
        ContinuationPoint<Field> make_point(
            const Field& field,
            const Model& model,
            double parameter,
            std::size_t corrector_steps,
            bool corrector_converged,
            const ContinuationOptions& options
        ) {
            const CoordinateSpace space(
                field, model, options.finite_difference_step
            );
            ContinuationPoint<Field> point(field);
            point.parameter = parameter;
            point.energy = model.energy(field);
            point.residual_norm = stability_detail::vector_norm(
                space.residual()
            ) / std::sqrt(static_cast<double>(space.size()));
            point.corrector_steps = corrector_steps;
            point.converged = corrector_converged
                && point.residual_norm <= options.corrector_tolerance;
            point.field = field;

            if (options.analyze_stability) {
                StabilityOptions stability_options = options.stability;
                stability_options.stationarity_tolerance = std::max(
                    stability_options.stationarity_tolerance,
                    options.corrector_tolerance
                );
                const auto stability = stability_analysis(
                    field, model, stability_options
                );
                if (!stability.eigenvalues.empty()) {
                    point.lowest_eigenvalue = stability.eigenvalues.front();
                }
                point.stable = stability.stable;
            }
            return point;
        }

        template <typename Field>
        void mark_bifurcation(
            ContinuationPoint<Field>& previous,
            ContinuationPoint<Field>& current,
            double tolerance
        ) {
            const double left = previous.lowest_eigenvalue;
            const double right = current.lowest_eigenvalue;
            if (!std::isfinite(left) || !std::isfinite(right)) {
                return;
            }
            const bool near_zero = std::abs(left) <= tolerance
                || std::abs(right) <= tolerance;
            const bool sign_change = (left < 0.0 && right > 0.0)
                || (left > 0.0 && right < 0.0);
            if (near_zero || sign_change) {
                previous.bifurcation_candidate = true;
                current.bifurcation_candidate = true;
            }
        }

        template <typename Field>
        struct CorrectorResult {
            Field field;
            double parameter{};
            double residual_norm{};
            std::size_t steps{};
            bool converged{};
        };

        template <
            typename Field,
            typename Model,
            typename Factory,
            typename CoordinateSpace
        >
        CorrectorResult<Field> correct(
            const Field& predictor,
            double predictor_parameter,
            const CoordinateVector& tangent_field,
            double tangent_parameter,
            const Factory& factory,
            const ContinuationOptions& options,
            unsigned int seed
        ) {
            Model predictor_model = factory(predictor_parameter);
            const CoordinateSpace chart(
                predictor,
                predictor_model,
                options.finite_difference_step
            );
            const std::size_t field_size = chart.size();
            CoordinateVector coordinates(field_size + 1, 0.0);

            const auto augmented_residual = [&](const CoordinateVector& value) {
                if (value.size() != field_size + 1) {
                    throw std::runtime_error(
                        "continuation coordinate size mismatch"
                    );
                }
                CoordinateVector field_coordinates(
                    value.begin(), value.begin() + field_size
                );
                const Field state = chart.retract(field_coordinates);
                const double parameter = predictor_parameter
                    + value.back() / options.parameter_scale;
                const Model model = factory(parameter);
                CoordinateVector result = chart.residual_at(state, model);
                result.push_back(
                    stability_detail::vector_dot(
                        tangent_field, field_coordinates
                    ) + tangent_parameter * value.back()
                );
                return result;
            };

            for (std::size_t step = 0;
                step <= options.max_corrector_steps; ++step) {
                const CoordinateVector current = augmented_residual(coordinates);
                const double norm = stability_detail::vector_norm(current)
                    / std::sqrt(static_cast<double>(current.size()));
                if (norm <= options.corrector_tolerance) {
                    CoordinateVector field_coordinates(
                        coordinates.begin(), coordinates.begin() + field_size
                    );
                    return {
                        chart.retract(field_coordinates),
                        predictor_parameter
                            + coordinates.back() / options.parameter_scale,
                        norm,
                        step,
                        true
                    };
                }
                if (step == options.max_corrector_steps) {
                    break;
                }

                const LinearOperator jacobian = [&](const CoordinateVector& direction) {
                    CoordinateVector plus = coordinates;
                    CoordinateVector minus = coordinates;
                    for (std::size_t index = 0;
                        index < direction.size(); ++index) {
                        plus[index] += options.finite_difference_step
                            * direction[index];
                        minus[index] -= options.finite_difference_step
                            * direction[index];
                    }
                    CoordinateVector plus_value = augmented_residual(plus);
                    const CoordinateVector minus_value = augmented_residual(minus);
                    const double scale = 0.5 / options.finite_difference_step;
                    for (std::size_t index = 0;
                        index < plus_value.size(); ++index) {
                        plus_value[index] = scale * (
                            plus_value[index] - minus_value[index]
                        );
                    }
                    return plus_value;
                };

                CoordinateVector right_hand_side = current;
                for (double& value : right_hand_side) {
                    value = -value;
                }
                const CoordinateVector inverse_diagonal =
                    stationary_detail::approximate_inverse_diagonal(
                        coordinates.size(),
                        jacobian,
                        options.preconditioner_probes,
                        options.preconditioner_floor,
                        seed + static_cast<unsigned int>(step)
                    );
                const LinearSolveResult linear = stationary_detail::gmres_impl(
                    jacobian,
                    right_hand_side,
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

                double damping = 1.0;
                CoordinateVector candidate = coordinates;
                stability_detail::axpy(candidate, damping, direction);
                double candidate_norm = stability_detail::vector_norm(
                    augmented_residual(candidate)
                ) / std::sqrt(static_cast<double>(candidate.size()));
                while ((!std::isfinite(candidate_norm)
                        || candidate_norm >= norm)
                    && damping > options.minimum_damping) {
                    damping *= 0.5;
                    candidate = coordinates;
                    stability_detail::axpy(candidate, damping, direction);
                    candidate_norm = stability_detail::vector_norm(
                        augmented_residual(candidate)
                    ) / std::sqrt(static_cast<double>(candidate.size()));
                }
                if (!std::isfinite(candidate_norm) || candidate_norm >= norm) {
                    break;
                }
                coordinates = std::move(candidate);
            }

            CoordinateVector field_coordinates(
                coordinates.begin(), coordinates.begin() + field_size
            );
            const CoordinateVector final_residual = augmented_residual(coordinates);
            return {
                chart.retract(field_coordinates),
                predictor_parameter
                    + coordinates.back() / options.parameter_scale,
                stability_detail::vector_norm(final_residual)
                    / std::sqrt(static_cast<double>(final_residual.size())),
                options.max_corrector_steps,
                false
            };
        }

        template <
            typename Field,
            typename Model,
            typename Factory,
            typename CoordinateSpace
        >
        BranchResult<Field> continue_impl(
            const Field& initial_field,
            const Factory& factory,
            const ContinuationOptions& options
        ) {
            validate(options);
            const double direction = options.stop >= options.start ? 1.0 : -1.0;
            double arc_step = std::min(
                options.maximum_step,
                std::max(options.minimum_step, std::abs(options.step))
            );

            BranchResult<Field> branch;
            Field first_field = initial_field;
            Model first_model = factory(options.start);
            StationaryOptions stationary_options = options.stationary;
            stationary_options.tolerance = std::min(
                stationary_options.tolerance,
                options.corrector_tolerance
            );
            auto first_history = solve_stationary(
                first_field, first_model, stationary_options
            );
            const bool first_converged = !first_history.empty()
                && first_history.back().converged;
            branch.points.push_back(make_point<Field, Model, CoordinateSpace>(
                first_field,
                first_model,
                options.start,
                0,
                first_converged,
                options
            ));
            if (!first_converged) {
                return branch;
            }
            if (reached(options.start, options.stop, direction)) {
                branch.reached_stop = true;
                branch.converged = true;
                return branch;
            }

            const double second_parameter = options.start + direction * std::min(
                arc_step,
                std::abs(options.stop - options.start)
            );
            Field second_field = first_field;
            Model second_model = factory(second_parameter);
            auto second_history = solve_stationary(
                second_field, second_model, stationary_options
            );
            const bool second_converged = !second_history.empty()
                && second_history.back().converged;
            branch.points.push_back(make_point<Field, Model, CoordinateSpace>(
                second_field,
                second_model,
                second_parameter,
                second_history.size(),
                second_converged,
                options
            ));
            mark_bifurcation(
                branch.points[0],
                branch.points[1],
                options.bifurcation_tolerance
            );
            if (!second_converged) {
                return branch;
            }
            if (reached(second_parameter, options.stop, direction)) {
                branch.reached_stop = true;
                branch.converged = true;
                return branch;
            }

            while (branch.points.size() < options.max_points) {
                const auto& previous = branch.points[branch.points.size() - 2];
                const auto& current = branch.points.back();
                const Model current_model = factory(current.parameter);
                const CoordinateSpace current_chart(
                    current.field,
                    current_model,
                    options.finite_difference_step
                );
                CoordinateVector tangent_field = current_chart.difference_to(
                    previous.field
                );
                for (double& value : tangent_field) {
                    value = -value;
                }
                double tangent_parameter = options.parameter_scale
                    * (current.parameter - previous.parameter);
                double tangent_norm = std::sqrt(
                    stability_detail::vector_dot(
                        tangent_field, tangent_field
                    ) + tangent_parameter * tangent_parameter
                );
                if (tangent_norm <= 1e-14) {
                    break;
                }
                for (double& value : tangent_field) {
                    value /= tangent_norm;
                }
                tangent_parameter /= tangent_norm;
                if (direction * tangent_parameter < 0.0
                    && branch.points.size() == 2) {
                    for (double& value : tangent_field) {
                        value = -value;
                    }
                    tangent_parameter = -tangent_parameter;
                }

                bool accepted = false;
                CorrectorResult<Field> corrected{
                    current.field,
                    current.parameter,
                    current.residual_norm,
                    0,
                    false
                };
                double attempted_step = arc_step;
                while (attempted_step >= options.minimum_step) {
                    const Field predictor = current_chart.retract(
                        tangent_field, attempted_step
                    );
                    const double predictor_parameter = current.parameter
                        + attempted_step * tangent_parameter
                            / options.parameter_scale;
                    corrected = correct<
                        Field,
                        Model,
                        Factory,
                        CoordinateSpace
                    >(
                        predictor,
                        predictor_parameter,
                        tangent_field,
                        tangent_parameter,
                        factory,
                        options,
                        options.seed + static_cast<unsigned int>(
                            branch.points.size() * 37
                        )
                    );
                    if (corrected.converged) {
                        accepted = true;
                        break;
                    }
                    attempted_step *= options.step_shrink;
                }
                if (!accepted) {
                    break;
                }

                const Model corrected_model = factory(corrected.parameter);
                branch.points.push_back(make_point<Field, Model, CoordinateSpace>(
                    corrected.field,
                    corrected_model,
                    corrected.parameter,
                    corrected.steps,
                    corrected.converged,
                    options
                ));
                mark_bifurcation(
                    branch.points[branch.points.size() - 2],
                    branch.points.back(),
                    options.bifurcation_tolerance
                );

                arc_step = attempted_step;
                if (corrected.steps <= options.target_corrector_steps) {
                    arc_step = std::min(
                        options.maximum_step,
                        arc_step * options.step_growth
                    );
                }
                if (reached(corrected.parameter, options.stop, direction)) {
                    branch.reached_stop = true;
                    break;
                }
            }
            branch.converged = branch.reached_stop
                && !branch.points.empty()
                && branch.points.back().converged;
            return branch;
        }

    } // namespace continuation_detail

    template <typename Factory>
    BranchResult<ScalarField2D> continue_solution(
        const ScalarField2D& field,
        const Factory& factory,
        const ContinuationOptions& options = {}
    ) {
        using Model = decltype(factory(options.start));
        return continuation_detail::continue_impl<
            ScalarField2D,
            Model,
            Factory,
            stability_detail::ScalarCoordinateSpace<ScalarField2D>
        >(field, factory, options);
    }

    template <typename Factory>
    BranchResult<XYField> continue_solution(
        const XYField& field,
        const Factory& factory,
        const ContinuationOptions& options = {}
    ) {
        using Model = decltype(factory(options.start));
        return continuation_detail::continue_impl<
            XYField,
            Model,
            Factory,
            stability_detail::ScalarCoordinateSpace<XYField>
        >(field, factory, options);
    }

    template <typename Factory>
    BranchResult<O3Field> continue_solution(
        const O3Field& field,
        const Factory& factory,
        const ContinuationOptions& options = {}
    ) {
        using Model = decltype(factory(options.start));
        return continuation_detail::continue_impl<
            O3Field,
            Model,
            Factory,
            stability_detail::SphereCoordinateSpace<O3Field>
        >(field, factory, options);
    }

    template <typename Factory>
    BranchResult<O3Field3D> continue_solution(
        const O3Field3D& field,
        const Factory& factory,
        const ContinuationOptions& options = {}
    ) {
        using Model = decltype(factory(options.start));
        return continuation_detail::continue_impl<
            O3Field3D,
            Model,
            Factory,
            stability_detail::SphereCoordinateSpace<O3Field3D>
        >(field, factory, options);
    }

} // namespace solitonkit
