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
#include "solitonkit/observables/GeometricTopologicalCharge.hpp"
#include "solitonkit/operators/DifferentialOperators.hpp"

namespace solitonkit {
namespace topology {

    constexpr double pi = 3.141592653589793238462643383279502884;

    inline double degree(const O3Field& field) {
        return GeometricTopologicalCharge::compute(field);
    }

    inline double wrapped_angle(double value) {
        return std::atan2(std::sin(value), std::cos(value));
    }

    struct VortexDefect {
        double x{};
        double y{};
        int charge{};
        std::size_t i{};
        std::size_t j{};
    };

    inline std::vector<VortexDefect> detect_defects(
        const XYField& field,
        double threshold = 0.5
    ) {
        if (threshold <= 0.0 || threshold > 1.0) {
            throw std::runtime_error("vortex threshold must be in (0, 1]");
        }
        const auto& lattice = field.lattice();
        if (lattice.nx() < 2 || lattice.ny() < 2) {
            throw std::runtime_error("vortex detection requires a 2x2 field");
        }
        const std::size_t count_x = lattice.boundary_x()
                == BoundaryCondition::Periodic
            ? lattice.nx() : lattice.nx() - 1;
        const std::size_t count_y = lattice.boundary_y()
                == BoundaryCondition::Periodic
            ? lattice.ny() : lattice.ny() - 1;
        const double center_x = 0.5 * static_cast<double>(lattice.nx() - 1);
        const double center_y = 0.5 * static_cast<double>(lattice.ny() - 1);

        std::vector<VortexDefect> result;
        for (std::size_t j = 0; j < count_y; ++j) {
            const std::size_t up = lattice.up(j);
            for (std::size_t i = 0; i < count_x; ++i) {
                const std::size_t right = lattice.right(i);
                const double circulation =
                    wrapped_angle(field(right, j) - field(i, j))
                    + wrapped_angle(field(right, up) - field(right, j))
                    + wrapped_angle(field(i, up) - field(right, up))
                    + wrapped_angle(field(i, j) - field(i, up));
                const double winding = circulation / (2.0 * pi);
                if (std::abs(winding) < threshold) {
                    continue;
                }
                const int charge = static_cast<int>(std::llround(winding));
                if (charge == 0) {
                    continue;
                }
                result.push_back({
                    (static_cast<double>(i) + 0.5 - center_x) * lattice.dx(),
                    (static_cast<double>(j) + 0.5 - center_y) * lattice.dy(),
                    charge,
                    i,
                    j
                });
            }
        }
        return result;
    }

    inline int vortex_number(const XYField& field) {
        int result = 0;
        for (const auto& defect : detect_defects(field)) {
            result += defect.charge;
        }
        return result;
    }

    inline int winding_number(const XYField& field) {
        return vortex_number(field);
    }

    struct HopfChargeOptions {
        std::size_t max_iterations{ 2000 };
        double tolerance{ 1e-8 };
    };

    struct HopfChargeResult {
        double charge{};
        double poisson_residual{};
        double divergence_norm{};
        std::size_t iterations{};
        bool converged{};
    };

    namespace detail {

        inline double vector_dot(
            const std::vector<Vec3>& left,
            const std::vector<Vec3>& right
        ) {
            if (left.size() != right.size()) {
                throw std::runtime_error("3D vector field sizes do not match");
            }
            double result = 0.0;
            for (std::size_t index = 0; index < left.size(); ++index) {
                result += dot(left[index], right[index]);
            }
            return result;
        }

        inline std::vector<Vec3> negative_laplacian_periodic(
            const std::vector<Vec3>& values,
            const Lattice3D& lattice
        ) {
            if (values.size() != lattice.size()) {
                throw std::runtime_error("3D vector field size mismatch");
            }
            std::vector<Vec3> result(values.size());
            const double inverse_dx2 = 1.0 / (lattice.dx() * lattice.dx());
            const double inverse_dy2 = 1.0 / (lattice.dy() * lattice.dy());
            const double inverse_dz2 = 1.0 / (lattice.dz() * lattice.dz());
            for (std::size_t k = 0; k < lattice.nz(); ++k) {
                const std::size_t back = (k + lattice.nz() - 1) % lattice.nz();
                const std::size_t front = (k + 1) % lattice.nz();
                for (std::size_t j = 0; j < lattice.ny(); ++j) {
                    const std::size_t down = (j + lattice.ny() - 1)
                        % lattice.ny();
                    const std::size_t up = (j + 1) % lattice.ny();
                    for (std::size_t i = 0; i < lattice.nx(); ++i) {
                        const std::size_t left = (i + lattice.nx() - 1)
                            % lattice.nx();
                        const std::size_t right = (i + 1) % lattice.nx();
                        const std::size_t index = lattice.index(i, j, k);
                        result[index] = inverse_dx2 * (
                                2.0 * values[index]
                                - values[lattice.index(left, j, k)]
                                - values[lattice.index(right, j, k)]
                            ) + inverse_dy2 * (
                                2.0 * values[index]
                                - values[lattice.index(i, down, k)]
                                - values[lattice.index(i, up, k)]
                            ) + inverse_dz2 * (
                                2.0 * values[index]
                                - values[lattice.index(i, j, back)]
                                - values[lattice.index(i, j, front)]
                            );
                    }
                }
            }
            return result;
        }

        inline std::vector<Vec3> curl_periodic(
            const std::vector<Vec3>& values,
            const Lattice3D& lattice
        ) {
            if (values.size() != lattice.size()) {
                throw std::runtime_error("3D vector field size mismatch");
            }
            std::vector<Vec3> result(values.size());
            for (std::size_t k = 0; k < lattice.nz(); ++k) {
                const std::size_t back = (k + lattice.nz() - 1) % lattice.nz();
                const std::size_t front = (k + 1) % lattice.nz();
                for (std::size_t j = 0; j < lattice.ny(); ++j) {
                    const std::size_t down = (j + lattice.ny() - 1)
                        % lattice.ny();
                    const std::size_t up = (j + 1) % lattice.ny();
                    for (std::size_t i = 0; i < lattice.nx(); ++i) {
                        const std::size_t left = (i + lattice.nx() - 1)
                            % lattice.nx();
                        const std::size_t right = (i + 1) % lattice.nx();
                        const Vec3 dx = (
                            values[lattice.index(right, j, k)]
                            - values[lattice.index(left, j, k)]
                        ) / (2.0 * lattice.dx());
                        const Vec3 dy = (
                            values[lattice.index(i, up, k)]
                            - values[lattice.index(i, down, k)]
                        ) / (2.0 * lattice.dy());
                        const Vec3 dz = (
                            values[lattice.index(i, j, front)]
                            - values[lattice.index(i, j, back)]
                        ) / (2.0 * lattice.dz());
                        result[lattice.index(i, j, k)] = {
                            dy.z - dz.y,
                            dz.x - dx.z,
                            dx.y - dy.x
                        };
                    }
                }
            }
            return result;
        }

        inline double divergence_norm_periodic(
            const std::vector<Vec3>& values,
            const Lattice3D& lattice
        ) {
            double squared = 0.0;
            for (std::size_t k = 0; k < lattice.nz(); ++k) {
                const std::size_t back = (k + lattice.nz() - 1) % lattice.nz();
                const std::size_t front = (k + 1) % lattice.nz();
                for (std::size_t j = 0; j < lattice.ny(); ++j) {
                    const std::size_t down = (j + lattice.ny() - 1)
                        % lattice.ny();
                    const std::size_t up = (j + 1) % lattice.ny();
                    for (std::size_t i = 0; i < lattice.nx(); ++i) {
                        const std::size_t left = (i + lattice.nx() - 1)
                            % lattice.nx();
                        const std::size_t right = (i + 1) % lattice.nx();
                        const double divergence = (
                            values[lattice.index(right, j, k)].x
                            - values[lattice.index(left, j, k)].x
                        ) / (2.0 * lattice.dx()) + (
                            values[lattice.index(i, up, k)].y
                            - values[lattice.index(i, down, k)].y
                        ) / (2.0 * lattice.dy()) + (
                            values[lattice.index(i, j, front)].z
                            - values[lattice.index(i, j, back)].z
                        ) / (2.0 * lattice.dz());
                        squared += divergence * divergence;
                    }
                }
            }
            return std::sqrt(squared / static_cast<double>(lattice.size()));
        }

    } // namespace detail

    inline HopfChargeResult hopf_charge(
        const O3Field3D& field,
        const HopfChargeOptions& options = {}
    ) {
        if (options.max_iterations == 0 || options.tolerance <= 0.0) {
            throw std::runtime_error("invalid Hopf charge solver options");
        }
        const auto& lattice = field.lattice();
        if (lattice.nx() < 3 || lattice.ny() < 3 || lattice.nz() < 3) {
            throw std::runtime_error(
                "Hopf charge requires at least a 3x3x3 field"
            );
        }

        std::vector<Vec3> magnetic(field.size());
        for (std::size_t k = 0; k < lattice.nz(); ++k) {
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    const Vec3 value = field(i, j, k);
                    const Vec3 dx = differential::derivative_x(field, i, j, k);
                    const Vec3 dy = differential::derivative_y(field, i, j, k);
                    const Vec3 dz = differential::derivative_z(field, i, j, k);
                    magnetic[lattice.index(i, j, k)] = {
                        dot(value, cross(dy, dz)),
                        dot(value, cross(dz, dx)),
                        dot(value, cross(dx, dy))
                    };
                }
            }
        }

        const std::vector<Vec3> right_hand_side = detail::curl_periodic(
            magnetic, lattice
        );
        std::vector<Vec3> potential(field.size());
        std::vector<Vec3> residual = right_hand_side;
        std::vector<Vec3> direction = residual;
        double residual_squared = detail::vector_dot(residual, residual);
        const double initial_norm = std::sqrt(residual_squared);
        const double target = options.tolerance * std::max(1.0, initial_norm);
        std::size_t iterations = 0;
        bool converged = initial_norm <= target;

        while (!converged && iterations < options.max_iterations) {
            const std::vector<Vec3> applied =
                detail::negative_laplacian_periodic(direction, lattice);
            const double denominator = detail::vector_dot(direction, applied);
            if (std::abs(denominator) <= 1e-30) {
                break;
            }
            const double alpha = residual_squared / denominator;
            for (std::size_t index = 0; index < field.size(); ++index) {
                potential[index] += alpha * direction[index];
                residual[index] -= alpha * applied[index];
            }
            const double next_squared = detail::vector_dot(residual, residual);
            ++iterations;
            if (std::sqrt(next_squared) <= target) {
                residual_squared = next_squared;
                converged = true;
                break;
            }
            const double beta = next_squared / residual_squared;
            for (std::size_t index = 0; index < field.size(); ++index) {
                direction[index] = residual[index] + beta * direction[index];
            }
            residual_squared = next_squared;
        }

        const double volume = lattice.dx() * lattice.dy() * lattice.dz();
        const double helicity = detail::vector_dot(potential, magnetic) * volume;
        const double charge = helicity / (16.0 * pi * pi);
        return {
            charge,
            std::sqrt(residual_squared) / std::max(1.0, initial_norm),
            detail::divergence_norm_periodic(magnetic, lattice),
            iterations,
            converged
        };
    }

} // namespace topology
} // namespace solitonkit
