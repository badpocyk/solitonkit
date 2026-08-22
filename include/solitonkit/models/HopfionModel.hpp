#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "solitonkit/core/O3Field3D.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/models/Model.hpp"
#include "solitonkit/operators/DifferentialOperators.hpp"

namespace solitonkit {

    struct HopfionEnergyTerms {
        double sigma{};
        double skyrme{};
        double potential{};

        double total() const { return sigma + skyrme + potential; }
    };

    class HopfionModel : public DifferentiableModel<O3Field3D, Vec3> {
    public:
        HopfionModel(
            double coupling = 1.0,
            double kappa = 1.0,
            double mass = 0.0
        ) : coupling_(coupling), kappa_(kappa), mass_(mass)
        {
            if (coupling_ <= 0.0 || kappa_ < 0.0 || mass_ < 0.0) {
                throw std::runtime_error(
                    "HopfionModel requires positive coupling and non-negative kappa/mass"
                );
            }
        }

        std::string name() const override { return "faddeev-skyrme-hopfion"; }
        std::size_t dimensions() const override { return 3; }
        FieldKind field_kind() const override { return FieldKind::O3_3D; }
        double coupling() const { return coupling_; }
        double kappa() const { return kappa_; }
        double mass() const { return mass_; }

        HopfionEnergyTerms energy_density_terms_at(
            const O3Field3D& field,
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            const Vec3 dx = differential::derivative_x(field, i, j, k);
            const Vec3 dy = differential::derivative_y(field, i, j, k);
            const Vec3 dz = differential::derivative_z(field, i, j, k);
            const double sigma = 0.5 * coupling_ * (
                dx.norm_squared() + dy.norm_squared() + dz.norm_squared()
            );
            const double skyrme = 0.5 * kappa_ * (
                cross(dx, dy).norm_squared()
                + cross(dx, dz).norm_squared()
                + cross(dy, dz).norm_squared()
            );
            const double potential = mass_ * mass_ * (1.0 - field(i, j, k).z);
            return { sigma, skyrme, potential };
        }

        HopfionEnergyTerms energy_terms(const O3Field3D& field) const {
            const auto& lattice = field.lattice();
            const double volume = lattice.dx() * lattice.dy() * lattice.dz();
            HopfionEnergyTerms result;
            for (std::size_t k = 0; k < lattice.nz(); ++k) {
                for (std::size_t j = 0; j < lattice.ny(); ++j) {
                    for (std::size_t i = 0; i < lattice.nx(); ++i) {
                        const auto terms = energy_density_terms_at(field, i, j, k);
                        result.sigma += volume * terms.sigma;
                        result.skyrme += volume * terms.skyrme;
                        result.potential += volume * terms.potential;
                    }
                }
            }
            return result;
        }

        double energy(const O3Field3D& field) const override {
            return energy_terms(field).total();
        }

        std::vector<Vec3> negative_gradient(
            const O3Field3D& field
        ) const override {
            const auto& lattice = field.lattice();
            std::vector<Vec3> result(field.size());
            for (std::size_t k = 0; k < lattice.nz(); ++k) {
                for (std::size_t j = 0; j < lattice.ny(); ++j) {
                    for (std::size_t i = 0; i < lattice.nx(); ++i) {
                        if (lattice.is_fixed_boundary(i, j, k)) {
                            continue;
                        }
                        const Vec3 skyrme_force = flux_divergence(
                            field, i, j, k
                        );
                        result[lattice.index(i, j, k)] =
                            coupling_ * differential::laplacian(field, i, j, k)
                            + skyrme_force
                            + Vec3{ 0.0, 0.0, mass_ * mass_ };
                    }
                }
            }
            return result;
        }

    private:
        Vec3 flux_x(
            const O3Field3D& field,
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            const Vec3 dx = differential::derivative_x(field, i, j, k);
            const Vec3 dy = differential::derivative_y(field, i, j, k);
            const Vec3 dz = differential::derivative_z(field, i, j, k);
            return kappa_ * (
                cross(dy, cross(dx, dy))
                + cross(dz, cross(dx, dz))
            );
        }

        Vec3 flux_y(
            const O3Field3D& field,
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            const Vec3 dx = differential::derivative_x(field, i, j, k);
            const Vec3 dy = differential::derivative_y(field, i, j, k);
            const Vec3 dz = differential::derivative_z(field, i, j, k);
            return kappa_ * (
                cross(cross(dx, dy), dx)
                + cross(dz, cross(dy, dz))
            );
        }

        Vec3 flux_z(
            const O3Field3D& field,
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            const Vec3 dx = differential::derivative_x(field, i, j, k);
            const Vec3 dy = differential::derivative_y(field, i, j, k);
            const Vec3 dz = differential::derivative_z(field, i, j, k);
            return kappa_ * (
                cross(cross(dx, dz), dx)
                + cross(cross(dy, dz), dy)
            );
        }

        Vec3 flux_divergence(
            const O3Field3D& field,
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            const auto& lattice = field.lattice();
            const Vec3 x = (
                flux_x(field, lattice.right(i), j, k)
                - flux_x(field, lattice.left(i), j, k)
            ) / (2.0 * lattice.dx());
            const Vec3 y = (
                flux_y(field, i, lattice.up(j), k)
                - flux_y(field, i, lattice.down(j), k)
            ) / (2.0 * lattice.dy());
            const Vec3 z = (
                flux_z(field, i, j, lattice.front(k))
                - flux_z(field, i, j, lattice.back(k))
            ) / (2.0 * lattice.dz());
            return x + y + z;
        }

        double coupling_{};
        double kappa_{};
        double mass_{};
    };

} // namespace solitonkit
