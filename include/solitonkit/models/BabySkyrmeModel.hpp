#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/models/Model.hpp"
#include "solitonkit/operators/DifferentialOperators.hpp"

namespace solitonkit {

    struct BabySkyrmeEnergyTerms {
        double sigma{};
        double skyrme{};
        double potential{};
        double dmi{};

        double total() const {
            return sigma + skyrme + potential + dmi;
        }
    };

    class BabySkyrmeModel : public DifferentiableModel<O3Field, Vec3> {
    public:
        BabySkyrmeModel(
            double kappa = 1.0,
            double mass = 1.0,
            double dmi = 0.0
        )
            : kappa_(kappa), mass_(mass), dmi_(dmi)
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

        double dmi() const {
            return dmi_;
        }

        std::string name() const override { return "baby-skyrme"; }
        std::size_t dimensions() const override { return 2; }
        FieldKind field_kind() const override { return FieldKind::O3_2D; }

        static Vec3 derivative_x(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            return differential::derivative_x(field, i, j);
        }

        static Vec3 derivative_y(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            return differential::derivative_y(field, i, j);
        }

        static Vec3 curl(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) {
            const Vec3 dx = derivative_x(field, i, j);
            const Vec3 dy = derivative_y(field, i, j);

            return {
                dy.z,
                -dx.z,
                dx.y - dy.x
            };
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

            const double dmi =
                dmi_ * dot(field(i, j), curl(field, i, j));

            return { sigma, skyrme, potential, dmi };
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
            double dmi = 0.0;

            const std::ptrdiff_t nx = static_cast<std::ptrdiff_t>(lat.nx());
            const std::ptrdiff_t ny = static_cast<std::ptrdiff_t>(lat.ny());

        #ifdef SOLITONKIT_USE_OPENMP
        #pragma omp parallel for reduction(+:sigma,skyrme,potential,dmi) collapse(2)
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
                    dmi += terms.dmi * cell_area;
                }
            }

            return { sigma, skyrme, potential, dmi };
        }

        double energy(const O3Field& field) const override {
            return energy_terms(field).total();
        }

        Vec3 skyrme_flux_x_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            const Vec3 dx = derivative_x(field, i, j);
            const Vec3 dy = derivative_y(field, i, j);
            return kappa_ * cross(dy, cross(dx, dy));
        }

        Vec3 skyrme_flux_y_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            const Vec3 dx = derivative_x(field, i, j);
            const Vec3 dy = derivative_y(field, i, j);
            return kappa_ * cross(cross(dx, dy), dx);
        }

        Vec3 negative_gradient_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            const auto& lattice = field.lattice();
            const Vec3 sigma = differential::laplacian(field, i, j);
            const Vec3 skyrme_x = (
                skyrme_flux_x_at(field, lattice.right(i), j)
                - skyrme_flux_x_at(field, lattice.left(i), j)
            ) / (2.0 * lattice.dx());
            const Vec3 skyrme_y = (
                skyrme_flux_y_at(field, i, lattice.up(j))
                - skyrme_flux_y_at(field, i, lattice.down(j))
            ) / (2.0 * lattice.dy());
            const Vec3 dmi_force = -2.0 * dmi_ * curl(field, i, j);
            const Vec3 potential_force{ 0.0, 0.0, mass_ * mass_ };
            return sigma + skyrme_x + skyrme_y
                + dmi_force + potential_force;
        }

        std::vector<Vec3> negative_gradient(
            const O3Field& field
        ) const override {
            const auto& lattice = field.lattice();
            std::vector<Vec3> result(field.size());
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    if (!lattice.is_fixed_boundary(i, j)) {
                        result[lattice.index(i, j)] =
                            negative_gradient_at(field, i, j);
                    }
                }
            }
            return result;
        }

    private:
        double kappa_;
        double mass_;
        double dmi_;
    };

} // namespace solitonkit
