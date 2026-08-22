#pragma once

#include <cstddef>
#include <type_traits>

#include "solitonkit/core/O3Field3D.hpp"
#include "solitonkit/core/ScalarField2D.hpp"
#include "solitonkit/core/Vec2.hpp"

namespace solitonkit::differential {

    template <typename Field>
    auto derivative_x(
        const Field& field,
        std::size_t i,
        std::size_t j
    ) -> std::decay_t<decltype(field(i, j))> {
        const auto& lattice = field.lattice();
        const auto zero = field(i, j) * 0.0;
        if (lattice.nx() == 1) {
            return zero;
        }

        if (lattice.boundary_x() != BoundaryCondition::Periodic) {
            if (lattice.boundary_x() == BoundaryCondition::Neumann
                && (i == 0 || i + 1 == lattice.nx())) {
                return zero;
            }
            if (i == 0) {
                return (field(1, j) - field(0, j)) / lattice.dx();
            }
            if (i + 1 == lattice.nx()) {
                return (field(i, j) - field(i - 1, j)) / lattice.dx();
            }
        }

        return (field(lattice.right(i), j) - field(lattice.left(i), j))
            / (2.0 * lattice.dx());
    }

    template <typename Field>
    auto derivative_y(
        const Field& field,
        std::size_t i,
        std::size_t j
    ) -> std::decay_t<decltype(field(i, j))> {
        const auto& lattice = field.lattice();
        const auto zero = field(i, j) * 0.0;
        if (lattice.ny() == 1) {
            return zero;
        }

        if (lattice.boundary_y() != BoundaryCondition::Periodic) {
            if (lattice.boundary_y() == BoundaryCondition::Neumann
                && (j == 0 || j + 1 == lattice.ny())) {
                return zero;
            }
            if (j == 0) {
                return (field(i, 1) - field(i, 0)) / lattice.dy();
            }
            if (j + 1 == lattice.ny()) {
                return (field(i, j) - field(i, j - 1)) / lattice.dy();
            }
        }

        return (field(i, lattice.up(j)) - field(i, lattice.down(j)))
            / (2.0 * lattice.dy());
    }

    template <typename Field>
    auto second_derivative_x(
        const Field& field,
        std::size_t i,
        std::size_t j
    ) -> std::decay_t<decltype(field(i, j))> {
        const auto& lattice = field.lattice();
        const auto center = field(i, j);
        if (lattice.nx() == 1) {
            return center * 0.0;
        }

        const auto left = field(lattice.left(i), j);
        const auto right = field(lattice.right(i), j);
        double ghost_factor = 1.0;
        if (lattice.boundary_x() == BoundaryCondition::Neumann
            && (i == 0 || i + 1 == lattice.nx())) {
            ghost_factor = 2.0;
        }
        return ghost_factor * (right - 2.0 * center + left)
            / (lattice.dx() * lattice.dx());
    }

    template <typename Field>
    auto second_derivative_y(
        const Field& field,
        std::size_t i,
        std::size_t j
    ) -> std::decay_t<decltype(field(i, j))> {
        const auto& lattice = field.lattice();
        const auto center = field(i, j);
        if (lattice.ny() == 1) {
            return center * 0.0;
        }

        const auto down = field(i, lattice.down(j));
        const auto up = field(i, lattice.up(j));
        double ghost_factor = 1.0;
        if (lattice.boundary_y() == BoundaryCondition::Neumann
            && (j == 0 || j + 1 == lattice.ny())) {
            ghost_factor = 2.0;
        }
        return ghost_factor * (up - 2.0 * center + down)
            / (lattice.dy() * lattice.dy());
    }

    template <typename Field>
    auto laplacian(
        const Field& field,
        std::size_t i,
        std::size_t j
    ) -> std::decay_t<decltype(field(i, j))> {
        return second_derivative_x(field, i, j)
            + second_derivative_y(field, i, j);
    }

    inline Vec2 gradient(
        const ScalarField2D& field,
        std::size_t i,
        std::size_t j
    ) {
        return {
            derivative_x(field, i, j),
            derivative_y(field, i, j)
        };
    }

    inline double divergence(
        const ScalarField2D& x_component,
        const ScalarField2D& y_component,
        std::size_t i,
        std::size_t j
    ) {
        return derivative_x(x_component, i, j)
            + derivative_y(y_component, i, j);
    }

    inline Vec3 curl(
        const O3Field& field,
        std::size_t i,
        std::size_t j
    ) {
        const Vec3 dx = derivative_x(field, i, j);
        const Vec3 dy = derivative_y(field, i, j);
        return { dy.z, -dx.z, dx.y - dy.x };
    }

    inline Vec3 derivative_x(
        const O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        const auto& lattice = field.lattice();
        if (lattice.nx() == 1) {
            return {};
        }
        if (lattice.boundary_x() != BoundaryCondition::Periodic) {
            if (lattice.boundary_x() == BoundaryCondition::Neumann
                && (i == 0 || i + 1 == lattice.nx())) {
                return {};
            }
            if (i == 0) {
                return (field(1, j, k) - field(0, j, k)) / lattice.dx();
            }
            if (i + 1 == lattice.nx()) {
                return (field(i, j, k) - field(i - 1, j, k)) / lattice.dx();
            }
        }
        return (field(lattice.right(i), j, k) - field(lattice.left(i), j, k))
            / (2.0 * lattice.dx());
    }

    inline Vec3 derivative_y(
        const O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        const auto& lattice = field.lattice();
        if (lattice.ny() == 1) {
            return {};
        }
        if (lattice.boundary_y() != BoundaryCondition::Periodic) {
            if (lattice.boundary_y() == BoundaryCondition::Neumann
                && (j == 0 || j + 1 == lattice.ny())) {
                return {};
            }
            if (j == 0) {
                return (field(i, 1, k) - field(i, 0, k)) / lattice.dy();
            }
            if (j + 1 == lattice.ny()) {
                return (field(i, j, k) - field(i, j - 1, k)) / lattice.dy();
            }
        }
        return (field(i, lattice.up(j), k) - field(i, lattice.down(j), k))
            / (2.0 * lattice.dy());
    }

    inline Vec3 derivative_z(
        const O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        const auto& lattice = field.lattice();
        if (lattice.nz() == 1) {
            return {};
        }
        if (lattice.boundary_z() != BoundaryCondition::Periodic) {
            if (lattice.boundary_z() == BoundaryCondition::Neumann
                && (k == 0 || k + 1 == lattice.nz())) {
                return {};
            }
            if (k == 0) {
                return (field(i, j, 1) - field(i, j, 0)) / lattice.dz();
            }
            if (k + 1 == lattice.nz()) {
                return (field(i, j, k) - field(i, j, k - 1)) / lattice.dz();
            }
        }
        return (field(i, j, lattice.front(k)) - field(i, j, lattice.back(k)))
            / (2.0 * lattice.dz());
    }

    inline Vec3 laplacian(
        const O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        const auto& lattice = field.lattice();
        const Vec3 center = field(i, j, k);
        Vec3 result{};
        if (lattice.nx() > 1) {
            const double factor = lattice.boundary_x() == BoundaryCondition::Neumann
                    && (i == 0 || i + 1 == lattice.nx())
                ? 2.0 : 1.0;
            result += factor * (field(lattice.right(i), j, k) - 2.0 * center
                + field(lattice.left(i), j, k))
                / (lattice.dx() * lattice.dx());
        }
        if (lattice.ny() > 1) {
            const double factor = lattice.boundary_y() == BoundaryCondition::Neumann
                    && (j == 0 || j + 1 == lattice.ny())
                ? 2.0 : 1.0;
            result += factor * (field(i, lattice.up(j), k) - 2.0 * center
                + field(i, lattice.down(j), k))
                / (lattice.dy() * lattice.dy());
        }
        if (lattice.nz() > 1) {
            const double factor = lattice.boundary_z() == BoundaryCondition::Neumann
                    && (k == 0 || k + 1 == lattice.nz())
                ? 2.0 : 1.0;
            result += factor * (field(i, j, lattice.front(k)) - 2.0 * center
                + field(i, j, lattice.back(k)))
                / (lattice.dz() * lattice.dz());
        }
        return result;
    }

    inline Vec3 curl(
        const O3Field3D& field,
        std::size_t i,
        std::size_t j,
        std::size_t k
    ) {
        const Vec3 dx = derivative_x(field, i, j, k);
        const Vec3 dy = derivative_y(field, i, j, k);
        const Vec3 dz = derivative_z(field, i, j, k);
        return {
            dy.z - dz.y,
            dz.x - dx.z,
            dx.y - dy.x
        };
    }

} // namespace solitonkit::differential
