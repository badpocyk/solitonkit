#pragma once

#include <cstddef>
#include <stdexcept>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    struct BabySkyrmeEnergyTerms {
        double sigma{};
        double skyrme{};
        double potential{};

        double total() const {
            return sigma + skyrme + potential;
        }
    };

    class BabySkyrmeModel {
    public:
        BabySkyrmeModel(double kappa = 1.0, double mass = 1.0)
            : kappa_(kappa), mass_(mass)
        {
            if (kappa_ < 0.0) {
                throw std::runtime_error("BabySkyrmeModel kappa must be non-negative");
            }

            if (mass_ < 0.0) {
                throw std::runtime_error("BabySkyrmeModel mass must be non-negative");
            }
        }

        double kappa() const {
            return kappa_;
        }

        double mass() const {
            return mass_;
        }

        static Vec3 derivative_x(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const std::size_t il = lat.left(i);
            const std::size_t ir = lat.right(i);

            return (field(ir, j) - field(il, j)) / (2.0 * lat.dx());
        }

        static Vec3 derivative_y(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const auto& lat = field.lattice();

            const std::size_t jd = lat.down(j);
            const std::size_t ju = lat.up(j);

            return (field(i, ju) - field(i, jd)) / (2.0 * lat.dy());
        }

        BabySkyrmeEnergyTerms energy_density_terms_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            const Vec3 dx = derivative_x(field, i, j);
            const Vec3 dy = derivative_y(field, i, j);

            const double sigma =
                0.5 * (dx.norm_squared() + dy.norm_squared());

            const Vec3 dx_cross_dy = cross(dx, dy);

            const double skyrme =
                0.5 * kappa_ * dx_cross_dy.norm_squared();

            const double potential =
                mass_ * mass_ * (1.0 - field(i, j).z);

            return { sigma, skyrme, potential };
        }

        double energy_density_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            return energy_density_terms_at(field, i, j).total();
        }

        BabySkyrmeEnergyTerms energy_terms(const O3Field& field) const {
            const auto& lat = field.lattice();

            if (lat.nx() < 3 || lat.ny() < 3) {
                throw std::runtime_error(
                    "BabySkyrmeModel energy requires at least a 3x3 field"
                );
            }

            const double cell_area = lat.dx() * lat.dy();

            double sigma = 0.0;
            double skyrme = 0.0;
            double potential = 0.0;

            const std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(lat.nx());
            const std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(lat.ny());

        #ifdef SOLITONKIT_USE_OPENMP
        #pragma omp parallel for reduction(+:sigma,skyrme,potential) collapse(2)
        #endif
            for (std::ptrdiff_t i = 0; i < nx; ++i) {
                for (std::ptrdiff_t j = 0; j < ny; ++j) {
                    const BabySkyrmeEnergyTerms terms = energy_density_terms_at(
                        field,
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j)
                    );

                    sigma += terms.sigma * cell_area;
                    skyrme += terms.skyrme * cell_area;
                    potential += terms.potential * cell_area;
                }
            }

            return { sigma, skyrme, potential };
        }

        double energy(const O3Field& field) const {
            return energy_terms(field).total();
        }

    private:
        double kappa_;
        double mass_;
    };

} // namespace solitonkit
