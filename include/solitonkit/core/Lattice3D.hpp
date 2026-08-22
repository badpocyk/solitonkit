#pragma once

#include <cstddef>
#include <stdexcept>

#include "solitonkit/core/BoundaryConditions.hpp"

namespace solitonkit {

    class Lattice3D {
    public:
        Lattice3D(
            std::size_t nx,
            std::size_t ny,
            std::size_t nz,
            double dx = 1.0,
            double dy = 1.0,
            double dz = 1.0,
            BoundaryCondition boundary_condition = BoundaryCondition::Periodic
        ) : Lattice3D(
                nx,
                ny,
                nz,
                dx,
                dy,
                dz,
                BoundaryConditions3D(boundary_condition)
            )
        {}

        Lattice3D(
            std::size_t nx,
            std::size_t ny,
            std::size_t nz,
            double dx,
            double dy,
            double dz,
            const BoundaryConditions3D& boundary_conditions
        ) : nx_(nx),
            ny_(ny),
            nz_(nz),
            dx_(dx),
            dy_(dy),
            dz_(dz),
            boundary_conditions_(boundary_conditions)
        {
            if (nx_ == 0 || ny_ == 0 || nz_ == 0) {
                throw std::runtime_error("Lattice dimensions must be positive");
            }
            if (dx_ <= 0.0 || dy_ <= 0.0 || dz_ <= 0.0) {
                throw std::runtime_error("Lattice spacings must be positive");
            }
        }

        std::size_t nx() const { return nx_; }
        std::size_t ny() const { return ny_; }
        std::size_t nz() const { return nz_; }
        double dx() const { return dx_; }
        double dy() const { return dy_; }
        double dz() const { return dz_; }
        std::size_t size() const { return nx_ * ny_ * nz_; }

        BoundaryCondition boundary_condition() const {
            return boundary_conditions_.x;
        }

        const BoundaryConditions3D& boundary_conditions() const {
            return boundary_conditions_;
        }

        BoundaryCondition boundary_x() const { return boundary_conditions_.x; }
        BoundaryCondition boundary_y() const { return boundary_conditions_.y; }
        BoundaryCondition boundary_z() const { return boundary_conditions_.z; }

        std::size_t index(std::size_t i, std::size_t j, std::size_t k) const {
            return map_boundary_index(
                    static_cast<std::ptrdiff_t>(i), nx_, boundary_conditions_.x
                )
                + nx_ * (
                    map_boundary_index(
                        static_cast<std::ptrdiff_t>(j), ny_, boundary_conditions_.y
                    )
                    + ny_ * map_boundary_index(
                        static_cast<std::ptrdiff_t>(k), nz_, boundary_conditions_.z
                    )
                );
        }

        std::size_t index_signed(
            std::ptrdiff_t i,
            std::ptrdiff_t j,
            std::ptrdiff_t k
        ) const {
            return map_boundary_index(i, nx_, boundary_conditions_.x)
                + nx_ * (
                    map_boundary_index(j, ny_, boundary_conditions_.y)
                    + ny_ * map_boundary_index(k, nz_, boundary_conditions_.z)
                );
        }

        std::size_t left(std::size_t i) const {
            return neighbor_minus(i, nx_, boundary_conditions_.x);
        }

        std::size_t right(std::size_t i) const {
            return neighbor_plus(i, nx_, boundary_conditions_.x);
        }

        std::size_t down(std::size_t j) const {
            return neighbor_minus(j, ny_, boundary_conditions_.y);
        }

        std::size_t up(std::size_t j) const {
            return neighbor_plus(j, ny_, boundary_conditions_.y);
        }

        std::size_t back(std::size_t k) const {
            return neighbor_minus(k, nz_, boundary_conditions_.z);
        }

        std::size_t front(std::size_t k) const {
            return neighbor_plus(k, nz_, boundary_conditions_.z);
        }

        bool is_boundary(std::size_t i, std::size_t j, std::size_t k) const {
            return i == 0 || i + 1 == nx_
                || j == 0 || j + 1 == ny_
                || k == 0 || k + 1 == nz_;
        }

        bool is_fixed_boundary(
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            return (pins_boundary(boundary_conditions_.x)
                    && (i == 0 || i + 1 == nx_))
                || (pins_boundary(boundary_conditions_.y)
                    && (j == 0 || j + 1 == ny_))
                || (pins_boundary(boundary_conditions_.z)
                    && (k == 0 || k + 1 == nz_));
        }

        bool is_dirichlet_boundary(
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            return (boundary_conditions_.x == BoundaryCondition::Dirichlet
                    && (i == 0 || i + 1 == nx_))
                || (boundary_conditions_.y == BoundaryCondition::Dirichlet
                    && (j == 0 || j + 1 == ny_))
                || (boundary_conditions_.z == BoundaryCondition::Dirichlet
                    && (k == 0 || k + 1 == nz_));
        }

    private:
        static std::size_t neighbor_minus(
            std::size_t value,
            std::size_t extent,
            BoundaryCondition condition
        ) {
            if (condition == BoundaryCondition::Periodic) {
                return (value + extent - 1) % extent;
            }
            return value == 0 ? 0 : value - 1;
        }

        static std::size_t neighbor_plus(
            std::size_t value,
            std::size_t extent,
            BoundaryCondition condition
        ) {
            if (condition == BoundaryCondition::Periodic) {
                return (value + 1) % extent;
            }
            return value + 1 < extent ? value + 1 : extent - 1;
        }

        std::size_t nx_{};
        std::size_t ny_{};
        std::size_t nz_{};
        double dx_{ 1.0 };
        double dy_{ 1.0 };
        double dz_{ 1.0 };
        BoundaryConditions3D boundary_conditions_{};
    };

} // namespace solitonkit
