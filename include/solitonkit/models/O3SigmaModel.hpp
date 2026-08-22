#pragma once

#include <cstddef>
#include <stdexcept>

#include "solitonkit/core/O3Field.hpp"
#include "solitonkit/core/Vec3.hpp"
#include "solitonkit/models/Model.hpp"
#include "solitonkit/operators/DifferentialOperators.hpp"

namespace solitonkit {

    class O3SigmaModel : public DifferentiableModel<O3Field, Vec3> {
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

        std::string name() const override { return "o3-sigma"; }
        std::size_t dimensions() const override { return 2; }
        FieldKind field_kind() const override { return FieldKind::O3_2D; }

        Vec3 derivative_x(const O3Field& field, std::size_t i, std::size_t j) const {
            return differential::derivative_x(field, i, j);
        }

        Vec3 derivative_y(const O3Field& field, std::size_t i, std::size_t j) const {
            return differential::derivative_y(field, i, j);
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

        double energy(const O3Field& field) const override {
            return total_energy(field);
        }

        std::vector<Vec3> negative_gradient(
            const O3Field& field
        ) const override {
            const auto& lattice = field.lattice();
            std::vector<Vec3> result(field.size());
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    if (!lattice.is_fixed_boundary(i, j)) {
                        result[lattice.index(i, j)] = coupling_
                            * differential::laplacian(field, i, j);
                    }
                }
            }
            return result;
        }

    private:
        double coupling_{ 1.0 };
    };

} // namespace solitonkit
