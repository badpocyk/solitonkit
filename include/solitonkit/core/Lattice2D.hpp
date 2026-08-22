#pragma once

#include <cstddef>
#include <stdexcept>

#include "solitonkit/core/BoundaryConditions.hpp"

namespace solitonkit {

    class Lattice2D {
    public:
        Lattice2D(
            std::size_t nx,
            std::size_t ny,
            double dx = 1.0,
            double dy = 1.0,
            BoundaryCondition boundary_condition = BoundaryCondition::Periodic
        )
            : Lattice2D(
                nx,
                ny,
                dx,
                dy,
                BoundaryConditions2D(boundary_condition)
            )
        {}

        Lattice2D(
            std::size_t nx,
            std::size_t ny,
            double dx,
            double dy,
            const BoundaryConditions2D& boundary_conditions
        )
            : nx_(nx),
            ny_(ny),
            dx_(dx),
            dy_(dy),
            boundary_conditions_(boundary_conditions)
        {
            if (nx_ == 0 || ny_ == 0) {
                throw std::runtime_error("Lattice dimensions must be positive");
            }

            if (dx_ <= 0.0 || dy_ <= 0.0) {
                throw std::runtime_error("Lattice spacings must be positive");
            }
        }

        std::size_t nx() const {
            return nx_;
        }

        std::size_t ny() const {
            return ny_;
        }

        double dx() const {
            return dx_;
        }

        double dy() const {
            return dy_;
        }

        BoundaryCondition boundary_condition() const {
            return boundary_conditions_.x;
        }

        const BoundaryConditions2D& boundary_conditions() const {
            return boundary_conditions_;
        }

        BoundaryCondition boundary_x() const {
            return boundary_conditions_.x;
        }

        BoundaryCondition boundary_y() const {
            return boundary_conditions_.y;
        }

        std::size_t size() const {
            return nx_ * ny_;
        }

        std::size_t index(std::size_t i, std::size_t j) const {
            return boundary_i(i) + nx_ * boundary_j(j);
        }

        std::size_t index_signed(std::ptrdiff_t i, std::ptrdiff_t j) const {
            return boundary_signed_i(i) + nx_ * boundary_signed_j(j);
        }

        std::size_t wrap_i(std::size_t i) const {
            return i % nx_;
        }

        std::size_t wrap_j(std::size_t j) const {
            return j % ny_;
        }

        std::size_t left(std::size_t i) const {
            if (boundary_conditions_.x == BoundaryCondition::Periodic) {
                return (i + nx_ - 1) % nx_;
            }

            return i == 0 ? 0 : i - 1;
        }

        std::size_t right(std::size_t i) const {
            if (boundary_conditions_.x == BoundaryCondition::Periodic) {
                return (i + 1) % nx_;
            }

            return i + 1 < nx_ ? i + 1 : nx_ - 1;
        }

        std::size_t down(std::size_t j) const {
            if (boundary_conditions_.y == BoundaryCondition::Periodic) {
                return (j + ny_ - 1) % ny_;
            }

            return j == 0 ? 0 : j - 1;
        }

        std::size_t up(std::size_t j) const {
            if (boundary_conditions_.y == BoundaryCondition::Periodic) {
                return (j + 1) % ny_;
            }

            return j + 1 < ny_ ? j + 1 : ny_ - 1;
        }

        bool is_boundary(std::size_t i, std::size_t j) const {
            return i == 0 || j == 0 || i + 1 == nx_ || j + 1 == ny_;
        }

        bool is_fixed_boundary(std::size_t i, std::size_t j) const {
            return (pins_boundary(boundary_conditions_.x)
                    && (i == 0 || i + 1 == nx_))
                || (pins_boundary(boundary_conditions_.y)
                    && (j == 0 || j + 1 == ny_));
        }

        bool is_dirichlet_boundary(std::size_t i, std::size_t j) const {
            return (boundary_conditions_.x == BoundaryCondition::Dirichlet
                    && (i == 0 || i + 1 == nx_))
                || (boundary_conditions_.y == BoundaryCondition::Dirichlet
                    && (j == 0 || j + 1 == ny_));
        }

    private:
        std::size_t boundary_i(std::size_t i) const {
            return map_boundary_index(
                static_cast<std::ptrdiff_t>(i),
                nx_,
                boundary_conditions_.x
            );
        }

        std::size_t boundary_j(std::size_t j) const {
            return map_boundary_index(
                static_cast<std::ptrdiff_t>(j),
                ny_,
                boundary_conditions_.y
            );
        }

        std::size_t boundary_signed_i(std::ptrdiff_t value) const {
            return map_boundary_index(value, nx_, boundary_conditions_.x);
        }

        std::size_t boundary_signed_j(std::ptrdiff_t value) const {
            return map_boundary_index(value, ny_, boundary_conditions_.y);
        }

    private:
        std::size_t nx_{};
        std::size_t ny_{};

        double dx_{ 1.0 };
        double dy_{ 1.0 };
        BoundaryConditions2D boundary_conditions_{};
    };

} // namespace solitonkit
