#pragma once

#include <cstddef>
#include <stdexcept>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    class O3SigmaModel {
    public:
        explicit O3SigmaModel(double coupling = 1.0)
            : coupling_(coupling)
        {
            if (coupling_ <= 0.0) {
                throw std::runtime_error("O3SigmaModel coupling must be positive");
            }
        }

        double coupling() const {
            return coupling_;
        }

        Vec3 derivative_x(const O3Field& field, std::size_t i, std::size_t j) const {
            const auto& lat = field.lattice();

            const std::size_t il = lat.left(i);
            const std::size_t ir = lat.right(i);

            return (field(ir, j) - field(il, j)) / (2.0 * lat.dx());
        }

        Vec3 derivative_y(const O3Field& field, std::size_t i, std::size_t j) const {
            const auto& lat = field.lattice();

            const std::size_t jd = lat.down(j);
            const std::size_t ju = lat.up(j);

            return (field(i, ju) - field(i, jd)) / (2.0 * lat.dy());
        }

        double energy_density_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            const Vec3 dx_phi = derivative_x(field, i, j);
            const Vec3 dy_phi = derivative_y(field, i, j);

            return 0.5 * coupling_ * (
                dot(dx_phi, dx_phi) + dot(dy_phi, dy_phi)
                );
        }

        double total_energy(const O3Field& field) const {
            const auto& lat = field.lattice();

            const double cell_area = lat.dx() * lat.dy();

            double energy = 0.0;

            const std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(lat.nx());
            const std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(lat.ny());

#ifdef SOLITONKIT_USE_OPENMP
#pragma omp parallel for reduction(+:energy) collapse(2)
#endif
            for (std::ptrdiff_t j = 0; j < ny; ++j) {
                for (std::ptrdiff_t i = 0; i < nx; ++i) {
                    energy += energy_density_at(
                        field,
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j)
                    ) * cell_area;
                }
            }

            return energy;
        }

    private:
        double coupling_{ 1.0 };
    };

} // namespace solitonkit