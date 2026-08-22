#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/O3Field3D.hpp"
#include "solitonkit/core/ScalarField2D.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/models/Model.hpp"

namespace solitonkit {

    struct StabilityOptions {
        std::size_t modes{ 6 };
        std::size_t max_iterations{ 80 };
        std::size_t subspace_dimension{};
        double tolerance{ 1e-7 };
        double finite_difference_step{ 1e-5 };
        double stationarity_tolerance{ 1e-6 };
        double eigenvalue_tolerance{ 1e-8 };
        unsigned int seed{ 12345 };
    };

    template <typename Value>
    struct StabilityResult {
        std::vector<double> eigenvalues;
        std::vector<double> residual_norms;
        std::vector<std::vector<Value>> modes;
        double gradient_norm{};
        std::size_t iterations{};
        std::size_t degrees_of_freedom{};
        bool converged{};
        bool stationary{};
        bool stable{};
    };

    namespace stability_detail {

        using CoordinateVector = std::vector<double>;
        using LinearOperator = std::function<CoordinateVector(
            const CoordinateVector&
        )>;

        inline double vector_dot(
            const CoordinateVector& left,
            const CoordinateVector& right
        ) {
            if (left.size() != right.size()) {
                throw std::runtime_error("coordinate vector sizes do not match");
            }
            return std::inner_product(
                left.begin(), left.end(), right.begin(), 0.0
            );
        }

        inline double vector_norm(const CoordinateVector& value) {
            return std::sqrt(std::max(0.0, vector_dot(value, value)));
        }

        inline void axpy(
            CoordinateVector& result,
            double scale,
            const CoordinateVector& value
        ) {
            if (result.size() != value.size()) {
                throw std::runtime_error("coordinate vector sizes do not match");
            }
            for (std::size_t index = 0; index < result.size(); ++index) {
                result[index] += scale * value[index];
            }
        }

        inline bool append_orthonormal(
            std::vector<CoordinateVector>& basis,
            CoordinateVector candidate,
            double threshold = 1e-12
        ) {
            for (int pass = 0; pass < 2; ++pass) {
                for (const auto& vector : basis) {
                    axpy(candidate, -vector_dot(candidate, vector), vector);
                }
            }
            const double norm = vector_norm(candidate);
            if (norm <= threshold) {
                return false;
            }
            for (double& value : candidate) {
                value /= norm;
            }
            basis.push_back(std::move(candidate));
            return true;
        }

        struct DenseEigensystem {
            std::vector<double> eigenvalues;
            std::vector<CoordinateVector> eigenvectors;
        };

        inline DenseEigensystem diagonalize_symmetric(
            std::vector<double> matrix,
            std::size_t size
        ) {
            if (matrix.size() != size * size) {
                throw std::runtime_error("invalid dense matrix size");
            }
            std::vector<double> eigenvectors(size * size, 0.0);
            for (std::size_t index = 0; index < size; ++index) {
                eigenvectors[index * size + index] = 1.0;
            }

            const std::size_t max_sweeps = std::max<std::size_t>(
                16, 8 * size * size
            );
            const double threshold = 32.0
                * std::numeric_limits<double>::epsilon();

            for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
                std::size_t p = 0;
                std::size_t q = 0;
                double largest = 0.0;
                for (std::size_t row = 0; row < size; ++row) {
                    for (std::size_t column = row + 1;
                        column < size; ++column) {
                        const double value = std::abs(
                            matrix[row * size + column]
                        );
                        if (value > largest) {
                            largest = value;
                            p = row;
                            q = column;
                        }
                    }
                }

                double diagonal_scale = 1.0;
                for (std::size_t index = 0; index < size; ++index) {
                    diagonal_scale = std::max(
                        diagonal_scale,
                        std::abs(matrix[index * size + index])
                    );
                }
                if (largest <= threshold * diagonal_scale) {
                    break;
                }

                const double app = matrix[p * size + p];
                const double aqq = matrix[q * size + q];
                const double apq = matrix[p * size + q];
                const double theta = 0.5 * (aqq - app) / apq;
                const double tangent = std::copysign(
                    1.0 / (
                        std::abs(theta)
                        + std::sqrt(1.0 + theta * theta)
                    ),
                    theta
                );
                const double cosine = 1.0 / std::sqrt(
                    1.0 + tangent * tangent
                );
                const double sine = tangent * cosine;

                for (std::size_t index = 0; index < size; ++index) {
                    if (index == p || index == q) {
                        continue;
                    }
                    const double aip = matrix[index * size + p];
                    const double aiq = matrix[index * size + q];
                    const double rotated_p = cosine * aip - sine * aiq;
                    const double rotated_q = sine * aip + cosine * aiq;
                    matrix[index * size + p] = rotated_p;
                    matrix[p * size + index] = rotated_p;
                    matrix[index * size + q] = rotated_q;
                    matrix[q * size + index] = rotated_q;
                }
                matrix[p * size + p] = app - tangent * apq;
                matrix[q * size + q] = aqq + tangent * apq;
                matrix[p * size + q] = 0.0;
                matrix[q * size + p] = 0.0;

                for (std::size_t row = 0; row < size; ++row) {
                    const double vip = eigenvectors[row * size + p];
                    const double viq = eigenvectors[row * size + q];
                    eigenvectors[row * size + p] =
                        cosine * vip - sine * viq;
                    eigenvectors[row * size + q] =
                        sine * vip + cosine * viq;
                }
            }

            std::vector<std::size_t> order(size);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](auto left, auto right) {
                return matrix[left * size + left]
                    < matrix[right * size + right];
            });

            DenseEigensystem result;
            result.eigenvalues.reserve(size);
            result.eigenvectors.reserve(size);
            for (const std::size_t column : order) {
                result.eigenvalues.push_back(
                    matrix[column * size + column]
                );
                CoordinateVector vector(size);
                for (std::size_t row = 0; row < size; ++row) {
                    vector[row] = eigenvectors[row * size + column];
                }
                result.eigenvectors.push_back(std::move(vector));
            }
            return result;
        }

        struct CoordinateStabilityResult {
            std::vector<double> eigenvalues;
            std::vector<double> residual_norms;
            std::vector<CoordinateVector> modes;
            std::size_t iterations{};
            bool converged{};
        };

        inline CoordinateVector combine(
            const std::vector<CoordinateVector>& basis,
            const CoordinateVector& coefficients
        ) {
            if (basis.empty() || basis.size() != coefficients.size()) {
                throw std::runtime_error("invalid Ritz vector coefficients");
            }
            CoordinateVector result(basis.front().size(), 0.0);
            for (std::size_t index = 0; index < basis.size(); ++index) {
                axpy(result, coefficients[index], basis[index]);
            }
            return result;
        }

        inline CoordinateStabilityResult lowest_eigenmodes(
            std::size_t degrees_of_freedom,
            const LinearOperator& apply,
            const StabilityOptions& options
        ) {
            if (options.modes == 0) {
                throw std::runtime_error("stability modes must be positive");
            }
            if (options.modes > degrees_of_freedom) {
                throw std::runtime_error(
                    "requested modes exceed active degrees of freedom"
                );
            }
            if (options.max_iterations == 0) {
                throw std::runtime_error(
                    "stability max_iterations must be positive"
                );
            }
            if (options.tolerance <= 0.0
                || options.finite_difference_step <= 0.0
                || options.stationarity_tolerance < 0.0
                || options.eigenvalue_tolerance < 0.0) {
                throw std::runtime_error(
                    "stability tolerances and finite-difference step are invalid"
                );
            }

            const std::size_t target = options.modes;
            std::size_t maximum_subspace = options.subspace_dimension;
            if (maximum_subspace == 0) {
                maximum_subspace = std::max<std::size_t>(
                    24, 4 * target + 8
                );
            }
            maximum_subspace = std::min(
                degrees_of_freedom,
                std::max(
                    std::min(degrees_of_freedom, target + 1),
                    maximum_subspace
                )
            );

            std::mt19937 generator(options.seed);
            std::normal_distribution<double> normal(0.0, 1.0);
            std::vector<CoordinateVector> basis;
            const std::size_t initial_size = std::min(
                maximum_subspace,
                std::max<std::size_t>(target + 2, 2 * target)
            );
            while (basis.size() < initial_size) {
                CoordinateVector candidate(degrees_of_freedom);
                for (double& value : candidate) {
                    value = normal(generator);
                }
                append_orthonormal(basis, std::move(candidate));
            }

            std::vector<CoordinateVector> applied;
            applied.reserve(maximum_subspace);
            for (const auto& vector : basis) {
                applied.push_back(apply(vector));
            }

            CoordinateStabilityResult result;
            for (std::size_t iteration = 1;
                iteration <= options.max_iterations; ++iteration) {
                const std::size_t subspace_size = basis.size();
                std::vector<double> projected(
                    subspace_size * subspace_size,
                    0.0
                );
                for (std::size_t row = 0; row < subspace_size; ++row) {
                    for (std::size_t column = row;
                        column < subspace_size; ++column) {
                        const double value = 0.5 * (
                            vector_dot(basis[row], applied[column])
                            + vector_dot(basis[column], applied[row])
                        );
                        projected[row * subspace_size + column] = value;
                        projected[column * subspace_size + row] = value;
                    }
                }

                const DenseEigensystem reduced = diagonalize_symmetric(
                    std::move(projected), subspace_size
                );
                result.eigenvalues.assign(
                    reduced.eigenvalues.begin(),
                    reduced.eigenvalues.begin() + target
                );
                result.residual_norms.assign(target, 0.0);
                result.modes.clear();
                result.modes.reserve(target);
                std::vector<CoordinateVector> residuals;
                residuals.reserve(target);

                bool all_converged = true;
                for (std::size_t mode = 0; mode < target; ++mode) {
                    CoordinateVector ritz = combine(
                        basis, reduced.eigenvectors[mode]
                    );
                    CoordinateVector applied_ritz = combine(
                        applied, reduced.eigenvectors[mode]
                    );
                    CoordinateVector residual = applied_ritz;
                    axpy(
                        residual,
                        -result.eigenvalues[mode],
                        ritz
                    );
                    const double residual_norm = vector_norm(residual);
                    result.residual_norms[mode] = residual_norm;
                    result.modes.push_back(std::move(ritz));
                    residuals.push_back(std::move(residual));

                    const double scaled_tolerance = options.tolerance
                        * std::max(1.0, std::abs(result.eigenvalues[mode]));
                    all_converged = all_converged
                        && residual_norm <= scaled_tolerance;
                }

                result.iterations = iteration;
                result.converged = all_converged;
                if (all_converged || subspace_size == degrees_of_freedom) {
                    return result;
                }

                const bool restart = basis.size() + target
                    > maximum_subspace;
                if (restart) {
                    std::vector<CoordinateVector> restarted;
                    restarted.reserve(maximum_subspace);
                    for (const auto& mode : result.modes) {
                        append_orthonormal(restarted, mode);
                    }
                    for (auto& residual : residuals) {
                        if (restarted.size() == maximum_subspace) {
                            break;
                        }
                        append_orthonormal(restarted, std::move(residual));
                    }
                    basis = std::move(restarted);
                    applied.clear();
                    for (const auto& vector : basis) {
                        applied.push_back(apply(vector));
                    }
                    continue;
                }

                bool expanded = false;
                for (auto& residual : residuals) {
                    if (basis.size() == maximum_subspace) {
                        break;
                    }
                    if (append_orthonormal(basis, std::move(residual))) {
                        applied.push_back(apply(basis.back()));
                        expanded = true;
                    }
                }
                if (!expanded) {
                    return result;
                }
            }
            return result;
        }

        template <typename Field>
        class ScalarCoordinateSpace {
        public:
            ScalarCoordinateSpace(
                const Field& field,
                const DifferentiableModel<Field, double>& model,
                double finite_difference_step
            ) : field_(field),
                model_(model),
                finite_difference_step_(finite_difference_step)
            {
                field_.enforce_boundary_condition();
                for (std::size_t index = 0; index < field_.size(); ++index) {
                    if (!fixed_at_index(index)) {
                        active_indices_.push_back(index);
                    }
                }
            }

            std::size_t size() const { return active_indices_.size(); }

            const Field& field() const { return field_; }

            Field retract(
                const CoordinateVector& direction,
                double scale = 1.0
            ) const {
                if (direction.size() != size()) {
                    throw std::runtime_error(
                        "scalar coordinate update has invalid size"
                    );
                }
                Field result = field_;
                for (std::size_t coordinate = 0;
                    coordinate < size(); ++coordinate) {
                    result.at_index(active_indices_[coordinate]) +=
                        scale * direction[coordinate];
                }
                result.enforce_boundary_condition();
                return result;
            }

            CoordinateVector difference_to(const Field& target) const {
                if (target.size() != field_.size()) {
                    throw std::runtime_error("scalar field sizes do not match");
                }
                CoordinateVector result(size());
                for (std::size_t coordinate = 0;
                    coordinate < size(); ++coordinate) {
                    const std::size_t index = active_indices_[coordinate];
                    result[coordinate] = target.at_index(index)
                        - field_.at_index(index);
                }
                return result;
            }

            CoordinateVector residual_at(
                const Field& state,
                const DifferentiableModel<Field, double>& model
            ) const {
                if (state.size() != field_.size()) {
                    throw std::runtime_error("scalar field sizes do not match");
                }
                return compress(model.negative_gradient(state));
            }

            CoordinateVector residual() const {
                return residual_at(field_, model_);
            }

            CoordinateVector apply(const CoordinateVector& direction) const {
                if (direction.size() != size()) {
                    throw std::runtime_error(
                        "Hessian direction size does not match scalar field"
                    );
                }
                Field plus = field_;
                Field minus = field_;
                for (std::size_t coordinate = 0;
                    coordinate < direction.size(); ++coordinate) {
                    const std::size_t index = active_indices_[coordinate];
                    const double perturbation = finite_difference_step_
                        * direction[coordinate];
                    plus.at_index(index) += perturbation;
                    minus.at_index(index) -= perturbation;
                }
                plus.enforce_boundary_condition();
                minus.enforce_boundary_condition();
                const auto plus_gradient = model_.negative_gradient(plus);
                const auto minus_gradient = model_.negative_gradient(minus);
                if (plus_gradient.size() != field_.size()
                    || minus_gradient.size() != field_.size()) {
                    throw std::runtime_error(
                        "model gradient size does not match scalar field"
                    );
                }

                CoordinateVector result(size());
                const double scale = 0.5 / finite_difference_step_;
                for (std::size_t coordinate = 0;
                    coordinate < size(); ++coordinate) {
                    const std::size_t index = active_indices_[coordinate];
                    result[coordinate] = scale * (
                        minus_gradient[index] - plus_gradient[index]
                    );
                }
                return result;
            }

            CoordinateVector compress(
                const std::vector<double>& values
            ) const {
                if (values.size() != field_.size()) {
                    throw std::runtime_error(
                        "scalar mode size does not match field"
                    );
                }
                CoordinateVector result(size());
                for (std::size_t coordinate = 0;
                    coordinate < size(); ++coordinate) {
                    result[coordinate] = values[active_indices_[coordinate]];
                }
                return result;
            }

            std::vector<double> expand(
                const CoordinateVector& values
            ) const {
                if (values.size() != size()) {
                    throw std::runtime_error(
                        "scalar coordinate mode has invalid size"
                    );
                }
                std::vector<double> result(field_.size(), 0.0);
                for (std::size_t coordinate = 0;
                    coordinate < size(); ++coordinate) {
                    result[active_indices_[coordinate]] = values[coordinate];
                }
                return result;
            }

            double gradient_norm() const {
                const auto gradient = model_.negative_gradient(field_);
                if (gradient.size() != field_.size()) {
                    throw std::runtime_error(
                        "model gradient size does not match scalar field"
                    );
                }
                double squared = 0.0;
                for (const std::size_t index : active_indices_) {
                    squared += gradient[index] * gradient[index];
                }
                return std::sqrt(
                    squared / std::max<std::size_t>(1, size())
                );
            }

        private:
            bool fixed_at_index(std::size_t index) const {
                const auto& lattice = field_.lattice();
                const std::size_t i = index % lattice.nx();
                const std::size_t j = index / lattice.nx();
                return lattice.is_fixed_boundary(i, j);
            }

            Field field_;
            const DifferentiableModel<Field, double>& model_;
            double finite_difference_step_{};
            std::vector<std::size_t> active_indices_;
        };

        inline Vec3 parallel_transport_to_base(
            const Vec3& value,
            const Vec3& point,
            const Vec3& base
        ) {
            const double denominator = 1.0 + dot(point, base);
            if (denominator <= 1e-12) {
                return value - dot(value, base) * base;
            }
            return value - (dot(value, base) / denominator) * (point + base);
        }

        template <typename Field>
        class SphereCoordinateSpace {
        public:
            SphereCoordinateSpace(
                const Field& field,
                const DifferentiableModel<Field, Vec3>& model,
                double finite_difference_step
            ) : field_(field),
                model_(model),
                finite_difference_step_(finite_difference_step)
            {
                field_.normalize_all();
                for (std::size_t index = 0; index < field_.size(); ++index) {
                    if (fixed_at_index(index)) {
                        continue;
                    }
                    active_indices_.push_back(index);
                    const Vec3 normal = field_.at_index(index);
                    const Vec3 reference = std::abs(normal.z) < 0.9
                        ? Vec3{ 0.0, 0.0, 1.0 }
                        : Vec3{ 1.0, 0.0, 0.0 };
                    const Vec3 first = cross(reference, normal).normalized();
                    basis_first_.push_back(first);
                    basis_second_.push_back(cross(normal, first).normalized());
                }
            }

            std::size_t size() const { return 2 * active_indices_.size(); }

            const Field& field() const { return field_; }

            Field retract(
                const CoordinateVector& direction,
                double scale = 1.0
            ) const {
                if (direction.size() != size()) {
                    throw std::runtime_error(
                        "O(3) coordinate update has invalid size"
                    );
                }
                Field result = field_;
                for (std::size_t site = 0;
                    site < active_indices_.size(); ++site) {
                    const Vec3 tangent = scale * (
                        direction[2 * site] * basis_first_[site]
                        + direction[2 * site + 1] * basis_second_[site]
                    );
                    const double angle = tangent.norm();
                    const Vec3 normal = field_.at_index(active_indices_[site]);
                    result.at_index(active_indices_[site]) = angle <= 1e-15
                        ? normal
                        : std::cos(angle) * normal
                            + (std::sin(angle) / angle) * tangent;
                }
                result.enforce_boundary_condition();
                return result;
            }

            CoordinateVector difference_to(const Field& target) const {
                if (target.size() != field_.size()) {
                    throw std::runtime_error("O(3) field sizes do not match");
                }
                std::vector<Vec3> logarithm(field_.size());
                for (std::size_t site = 0;
                    site < active_indices_.size(); ++site) {
                    const std::size_t index = active_indices_[site];
                    const Vec3 base = field_.at_index(index);
                    const Vec3 destination = target.at_index(index).normalized();
                    const double cosine = std::max(
                        -1.0, std::min(1.0, dot(base, destination))
                    );
                    const double angle = std::acos(cosine);
                    if (angle <= 1e-12) {
                        logarithm[index] = {};
                    }
                    else {
                        const Vec3 tangent = destination - cosine * base;
                        const double sine = std::sin(angle);
                        logarithm[index] = sine <= 1e-12
                            ? tangent
                            : (angle / sine) * tangent;
                    }
                }
                return compress(logarithm);
            }

            CoordinateVector residual_at(
                const Field& state,
                const DifferentiableModel<Field, Vec3>& model
            ) const {
                if (state.size() != field_.size()) {
                    throw std::runtime_error("O(3) field sizes do not match");
                }
                const auto gradient = model.negative_gradient(state);
                if (gradient.size() != field_.size()) {
                    throw std::runtime_error(
                        "model gradient size does not match O(3) field"
                    );
                }
                CoordinateVector result(size());
                for (std::size_t site = 0;
                    site < active_indices_.size(); ++site) {
                    const std::size_t index = active_indices_[site];
                    const Vec3 point = state.at_index(index).normalized();
                    const Vec3 tangent = gradient[index]
                        - dot(gradient[index], point) * point;
                    const Vec3 transported = parallel_transport_to_base(
                        tangent, point, field_.at_index(index)
                    );
                    result[2 * site] = dot(
                        transported, basis_first_[site]
                    );
                    result[2 * site + 1] = dot(
                        transported, basis_second_[site]
                    );
                }
                return result;
            }

            CoordinateVector residual() const {
                return residual_at(field_, model_);
            }

            CoordinateVector apply(const CoordinateVector& direction) const {
                if (direction.size() != size()) {
                    throw std::runtime_error(
                        "Hessian direction size does not match O(3) field"
                    );
                }
                Field plus = field_;
                Field minus = field_;
                for (std::size_t site = 0;
                    site < active_indices_.size(); ++site) {
                    const Vec3 tangent =
                        direction[2 * site] * basis_first_[site]
                        + direction[2 * site + 1] * basis_second_[site];
                    const double norm = tangent.norm();
                    const Vec3 normal = field_.at_index(active_indices_[site]);
                    if (norm <= 1e-15) {
                        plus.at_index(active_indices_[site]) = normal;
                        minus.at_index(active_indices_[site]) = normal;
                        continue;
                    }
                    const double angle = finite_difference_step_ * norm;
                    const Vec3 unit_tangent = tangent / norm;
                    plus.at_index(active_indices_[site]) =
                        std::cos(angle) * normal
                        + std::sin(angle) * unit_tangent;
                    minus.at_index(active_indices_[site]) =
                        std::cos(angle) * normal
                        - std::sin(angle) * unit_tangent;
                }
                plus.enforce_boundary_condition();
                minus.enforce_boundary_condition();
                const auto plus_gradient = model_.negative_gradient(plus);
                const auto minus_gradient = model_.negative_gradient(minus);
                if (plus_gradient.size() != field_.size()
                    || minus_gradient.size() != field_.size()) {
                    throw std::runtime_error(
                        "model gradient size does not match O(3) field"
                    );
                }

                CoordinateVector result(size());
                const double scale = 0.5 / finite_difference_step_;
                for (std::size_t site = 0;
                    site < active_indices_.size(); ++site) {
                    const std::size_t index = active_indices_[site];
                    const Vec3 plus_point = plus.at_index(index);
                    const Vec3 minus_point = minus.at_index(index);
                    const Vec3 plus_tangent = plus_gradient[index]
                        - dot(plus_gradient[index], plus_point) * plus_point;
                    const Vec3 minus_tangent = minus_gradient[index]
                        - dot(minus_gradient[index], minus_point) * minus_point;
                    const Vec3 transported_plus = parallel_transport_to_base(
                        plus_tangent, plus_point, field_.at_index(index)
                    );
                    const Vec3 transported_minus = parallel_transport_to_base(
                        minus_tangent, minus_point, field_.at_index(index)
                    );
                    const Vec3 hessian_value = scale * (
                        transported_minus - transported_plus
                    );
                    result[2 * site] = dot(
                        hessian_value, basis_first_[site]
                    );
                    result[2 * site + 1] = dot(
                        hessian_value, basis_second_[site]
                    );
                }
                return result;
            }

            CoordinateVector compress(
                const std::vector<Vec3>& values
            ) const {
                if (values.size() != field_.size()) {
                    throw std::runtime_error("O(3) mode size does not match field");
                }
                CoordinateVector result(size());
                for (std::size_t site = 0;
                    site < active_indices_.size(); ++site) {
                    const Vec3 tangent = values[active_indices_[site]]
                        - dot(
                            values[active_indices_[site]],
                            field_.at_index(active_indices_[site])
                        ) * field_.at_index(active_indices_[site]);
                    result[2 * site] = dot(tangent, basis_first_[site]);
                    result[2 * site + 1] = dot(
                        tangent, basis_second_[site]
                    );
                }
                return result;
            }

            std::vector<Vec3> expand(
                const CoordinateVector& values
            ) const {
                if (values.size() != size()) {
                    throw std::runtime_error(
                        "O(3) coordinate mode has invalid size"
                    );
                }
                std::vector<Vec3> result(field_.size());
                for (std::size_t site = 0;
                    site < active_indices_.size(); ++site) {
                    result[active_indices_[site]] =
                        values[2 * site] * basis_first_[site]
                        + values[2 * site + 1] * basis_second_[site];
                }
                return result;
            }

            double gradient_norm() const {
                const auto gradient = model_.negative_gradient(field_);
                if (gradient.size() != field_.size()) {
                    throw std::runtime_error(
                        "model gradient size does not match O(3) field"
                    );
                }
                double squared = 0.0;
                for (const std::size_t index : active_indices_) {
                    const Vec3 normal = field_.at_index(index);
                    const Vec3 tangent = gradient[index]
                        - dot(gradient[index], normal) * normal;
                    squared += tangent.norm_squared();
                }
                return std::sqrt(
                    squared / std::max<std::size_t>(1, active_indices_.size())
                );
            }

        private:
            bool fixed_at_index(std::size_t index) const {
                const auto& lattice = field_.lattice();
                if constexpr (std::is_same<Field, O3Field>::value) {
                    const std::size_t i = index % lattice.nx();
                    const std::size_t j = index / lattice.nx();
                    return lattice.is_fixed_boundary(i, j);
                }
                else {
                    const std::size_t plane = lattice.nx() * lattice.ny();
                    const std::size_t k = index / plane;
                    const std::size_t remainder = index % plane;
                    const std::size_t j = remainder / lattice.nx();
                    const std::size_t i = remainder % lattice.nx();
                    return lattice.is_fixed_boundary(i, j, k);
                }
            }

            Field field_;
            const DifferentiableModel<Field, Vec3>& model_;
            double finite_difference_step_{};
            std::vector<std::size_t> active_indices_;
            std::vector<Vec3> basis_first_;
            std::vector<Vec3> basis_second_;
        };

        template <typename Value, typename CoordinateSpace>
        StabilityResult<Value> analyze(
            const CoordinateSpace& space,
            const StabilityOptions& options
        ) {
            if (space.size() == 0) {
                throw std::runtime_error(
                    "stability analysis requires active degrees of freedom"
                );
            }
            const auto coordinate_result = lowest_eigenmodes(
                space.size(),
                [&](const CoordinateVector& direction) {
                    return space.apply(direction);
                },
                options
            );

            StabilityResult<Value> result;
            result.eigenvalues = coordinate_result.eigenvalues;
            result.residual_norms = coordinate_result.residual_norms;
            result.iterations = coordinate_result.iterations;
            result.degrees_of_freedom = space.size();
            result.converged = coordinate_result.converged;
            result.gradient_norm = space.gradient_norm();
            result.stationary = result.gradient_norm
                <= options.stationarity_tolerance;
            result.stable = result.stationary && result.converged
                && !result.eigenvalues.empty()
                && result.eigenvalues.front()
                    >= -options.eigenvalue_tolerance;
            result.modes.reserve(coordinate_result.modes.size());
            for (const auto& mode : coordinate_result.modes) {
                result.modes.push_back(space.expand(mode));
            }
            return result;
        }

    } // namespace stability_detail

    template <typename Field>
    StabilityResult<double> stability_analysis_scalar(
        const Field& field,
        const DifferentiableModel<Field, double>& model,
        const StabilityOptions& options = {}
    ) {
        const stability_detail::ScalarCoordinateSpace<Field> space(
            field, model, options.finite_difference_step
        );
        return stability_detail::analyze<double>(space, options);
    }

    inline StabilityResult<double> stability_analysis(
        const ScalarField2D& field,
        const DifferentiableModel<ScalarField2D, double>& model,
        const StabilityOptions& options = {}
    ) {
        return stability_analysis_scalar(field, model, options);
    }

    inline StabilityResult<double> stability_analysis(
        const XYField& field,
        const DifferentiableModel<XYField, double>& model,
        const StabilityOptions& options = {}
    ) {
        return stability_analysis_scalar(field, model, options);
    }

    template <typename Field>
    StabilityResult<Vec3> stability_analysis_sphere(
        const Field& field,
        const DifferentiableModel<Field, Vec3>& model,
        const StabilityOptions& options = {}
    ) {
        const stability_detail::SphereCoordinateSpace<Field> space(
            field, model, options.finite_difference_step
        );
        return stability_detail::analyze<Vec3>(space, options);
    }

    inline StabilityResult<Vec3> stability_analysis(
        const O3Field& field,
        const DifferentiableModel<O3Field, Vec3>& model,
        const StabilityOptions& options = {}
    ) {
        return stability_analysis_sphere(field, model, options);
    }

    inline StabilityResult<Vec3> stability_analysis(
        const O3Field3D& field,
        const DifferentiableModel<O3Field3D, Vec3>& model,
        const StabilityOptions& options = {}
    ) {
        return stability_analysis_sphere(field, model, options);
    }

    template <typename Field>
    std::vector<double> hessian_vector_product_scalar(
        const Field& field,
        const DifferentiableModel<Field, double>& model,
        const std::vector<double>& direction,
        double finite_difference_step = 1e-5
    ) {
        if (finite_difference_step <= 0.0) {
            throw std::runtime_error(
                "Hessian finite-difference step must be positive"
            );
        }
        const stability_detail::ScalarCoordinateSpace<Field> space(
            field, model, finite_difference_step
        );
        return space.expand(space.apply(space.compress(direction)));
    }

    inline std::vector<double> hessian_vector_product(
        const ScalarField2D& field,
        const DifferentiableModel<ScalarField2D, double>& model,
        const std::vector<double>& direction,
        double finite_difference_step = 1e-5
    ) {
        return hessian_vector_product_scalar(
            field, model, direction, finite_difference_step
        );
    }

    inline std::vector<double> hessian_vector_product(
        const XYField& field,
        const DifferentiableModel<XYField, double>& model,
        const std::vector<double>& direction,
        double finite_difference_step = 1e-5
    ) {
        return hessian_vector_product_scalar(
            field, model, direction, finite_difference_step
        );
    }

    template <typename Field>
    std::vector<Vec3> hessian_vector_product_sphere(
        const Field& field,
        const DifferentiableModel<Field, Vec3>& model,
        const std::vector<Vec3>& direction,
        double finite_difference_step = 1e-5
    ) {
        if (finite_difference_step <= 0.0) {
            throw std::runtime_error(
                "Hessian finite-difference step must be positive"
            );
        }
        const stability_detail::SphereCoordinateSpace<Field> space(
            field, model, finite_difference_step
        );
        return space.expand(space.apply(space.compress(direction)));
    }

    inline std::vector<Vec3> hessian_vector_product(
        const O3Field& field,
        const DifferentiableModel<O3Field, Vec3>& model,
        const std::vector<Vec3>& direction,
        double finite_difference_step = 1e-5
    ) {
        return hessian_vector_product_sphere(
            field, model, direction, finite_difference_step
        );
    }

    inline std::vector<Vec3> hessian_vector_product(
        const O3Field3D& field,
        const DifferentiableModel<O3Field3D, Vec3>& model,
        const std::vector<Vec3>& direction,
        double finite_difference_step = 1e-5
    ) {
        return hessian_vector_product_sphere(
            field, model, direction, finite_difference_step
        );
    }

} // namespace solitonkit
