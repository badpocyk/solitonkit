#pragma once

#include <cstddef>
#include <stdexcept>

namespace solitonkit {

    class Lattice2D {
    public:
        Lattice2D(
            std::size_t nx,
            std::size_t ny,
            double dx = 1.0,
            double dy = 1.0
        )
            : nx_(nx), ny_(ny), dx_(dx), dy_(dy)
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

        std::size_t size() const {
            return nx_ * ny_;
        }

        std::size_t index(std::size_t i, std::size_t j) const {
            return wrap_i(i) + nx_ * wrap_j(j);
        }

        std::size_t index_signed(std::ptrdiff_t i, std::ptrdiff_t j) const {
            return wrap_signed(i, nx_) + nx_ * wrap_signed(j, ny_);
        }

        std::size_t wrap_i(std::size_t i) const {
            return i % nx_;
        }

        std::size_t wrap_j(std::size_t j) const {
            return j % ny_;
        }

        std::size_t left(std::size_t i) const {
            return (i + nx_ - 1) % nx_;
        }

        std::size_t right(std::size_t i) const {
            return (i + 1) % nx_;
        }

        std::size_t down(std::size_t j) const {
            return (j + ny_ - 1) % ny_;
        }

        std::size_t up(std::size_t j) const {
            return (j + 1) % ny_;
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

    private:
        std::size_t nx_{};
        std::size_t ny_{};

        double dx_{ 1.0 };
        double dy_{ 1.0 };
    };

} // namespace solitonkit