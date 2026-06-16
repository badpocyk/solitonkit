#pragma once

#include <cstddef>
#include <stdexcept>

namespace solitonkit {

    enum class BoundaryCondition {
        Periodic,
        Fixed,
        Neumann,
        Dirichlet
    };

    class Lattice2D {
    public:
        Lattice2D(
            std::size_t nx,
            std::size_t ny,
            double dx = 1.0,
            double dy = 1.0,
            BoundaryCondition boundary_condition = BoundaryCondition::Periodic
        )
            : nx_(nx),
            ny_(ny),
            dx_(dx),
            dy_(dy),
            boundary_condition_(boundary_condition)
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
            return boundary_condition_;
        }

        std::size_t size() const {
            return nx_ * ny_;
        }

        std::size_t index(std::size_t i, std::size_t j) const {
            return boundary_i(i) + nx_ * boundary_j(j);
        }

        std::size_t index_signed(std::ptrdiff_t i, std::ptrdiff_t j) const {
            return boundary_signed(i, nx_) + nx_ * boundary_signed(j, ny_);
        }

        std::size_t wrap_i(std::size_t i) const {
            return i % nx_;
        }

        std::size_t wrap_j(std::size_t j) const {
            return j % ny_;
        }

        std::size_t left(std::size_t i) const {
            if (boundary_condition_ == BoundaryCondition::Periodic) {
                return (i + nx_ - 1) % nx_;
            }

            return i == 0 ? 0 : i - 1;
        }

        std::size_t right(std::size_t i) const {
            if (boundary_condition_ == BoundaryCondition::Periodic) {
                return (i + 1) % nx_;
            }

            return i + 1 < nx_ ? i + 1 : nx_ - 1;
        }

        std::size_t down(std::size_t j) const {
            if (boundary_condition_ == BoundaryCondition::Periodic) {
                return (j + ny_ - 1) % ny_;
            }

            return j == 0 ? 0 : j - 1;
        }

        std::size_t up(std::size_t j) const {
            if (boundary_condition_ == BoundaryCondition::Periodic) {
                return (j + 1) % ny_;
            }

            return j + 1 < ny_ ? j + 1 : ny_ - 1;
        }

        bool is_boundary(std::size_t i, std::size_t j) const {
            return i == 0 || j == 0 || i + 1 == nx_ || j + 1 == ny_;
        }

        bool is_fixed_boundary(std::size_t i, std::size_t j) const {
            return (
                boundary_condition_ == BoundaryCondition::Fixed
                || boundary_condition_ == BoundaryCondition::Dirichlet
            )
                && is_boundary(i, j);
        }

        bool is_dirichlet_boundary(std::size_t i, std::size_t j) const {
            return boundary_condition_ == BoundaryCondition::Dirichlet
                && is_boundary(i, j);
        }

    private:
        static std::size_t wrap_signed(std::ptrdiff_t value, std::size_t n) {
            const auto n_signed = static_cast<std::ptrdiff_t>(n);

            auto result = value % n_signed;

            if (result < 0) {
                result += n_signed;
            }

            return static_cast<std::size_t>(result);
        }

        std::size_t boundary_i(std::size_t i) const {
            if (boundary_condition_ == BoundaryCondition::Periodic) {
                return wrap_i(i);
            }

            return i < nx_ ? i : nx_ - 1;
        }

        std::size_t boundary_j(std::size_t j) const {
            if (boundary_condition_ == BoundaryCondition::Periodic) {
                return wrap_j(j);
            }

            return j < ny_ ? j : ny_ - 1;
        }

        std::size_t boundary_signed(std::ptrdiff_t value, std::size_t n) const {
            if (boundary_condition_ == BoundaryCondition::Periodic) {
                return wrap_signed(value, n);
            }

            if (value < 0) {
                return 0;
            }

            const auto result = static_cast<std::size_t>(value);

            return result < n ? result : n - 1;
        }

    private:
        std::size_t nx_{};
        std::size_t ny_{};

        double dx_{ 1.0 };
        double dy_{ 1.0 };
        BoundaryCondition boundary_condition_{ BoundaryCondition::Periodic };
    };

} // namespace solitonkit
