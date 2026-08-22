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

    enum class DMIType {
        None,
        Bulk,
        Interfacial
    };

    struct MicromagneticEnergyTerms {
        double exchange{};
        double dmi{};
        double anisotropy{};
        double zeeman{};

        double total() const {
            return exchange + dmi + anisotropy + zeeman;
        }
    };

    class MicromagneticModel : public DifferentiableModel<O3Field, Vec3> {
    public:
        MicromagneticModel(
            double exchange = 1.0,
            double dmi = 0.0,
            double anisotropy = 0.0,
            const Vec3& applied_field = Vec3{},
            const Vec3& easy_axis = Vec3{ 0.0, 0.0, 1.0 },
            DMIType dmi_type = DMIType::Bulk
        ) : exchange_(exchange),
            dmi_(dmi),
            anisotropy_(anisotropy),
            applied_field_(applied_field),
            easy_axis_(easy_axis.normalized()),
            dmi_type_(dmi_type)
        {
            if (exchange_ <= 0.0) {
                throw std::runtime_error(
                    "MicromagneticModel exchange must be positive"
                );
            }
            if (anisotropy_ < 0.0) {
                throw std::runtime_error(
                    "MicromagneticModel anisotropy must be non-negative"
                );
            }
        }

        std::string name() const override { return "micromagnetic"; }
        std::size_t dimensions() const override { return 2; }
        FieldKind field_kind() const override { return FieldKind::O3_2D; }

        double exchange() const { return exchange_; }
        double dmi() const { return dmi_; }
        double anisotropy() const { return anisotropy_; }
        const Vec3& applied_field() const { return applied_field_; }
        const Vec3& easy_axis() const { return easy_axis_; }
        DMIType dmi_type() const { return dmi_type_; }

        MicromagneticEnergyTerms energy_density_terms_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            const Vec3 m = field(i, j);
            const Vec3 dx = differential::derivative_x(field, i, j);
            const Vec3 dy = differential::derivative_y(field, i, j);
            const double exchange = exchange_
                * (dx.norm_squared() + dy.norm_squared());

            double dmi_energy = 0.0;
            if (dmi_type_ == DMIType::Bulk) {
                dmi_energy = dmi_ * dot(
                    m,
                    differential::curl(field, i, j)
                );
            }
            else if (dmi_type_ == DMIType::Interfacial) {
                dmi_energy = dmi_ * (
                    m.z * (dx.x + dy.y) - m.x * dx.z - m.y * dy.z
                );
            }

            const double alignment = dot(m, easy_axis_);
            const double anisotropy = anisotropy_
                * (1.0 - alignment * alignment);
            const double zeeman = -dot(applied_field_, m);
            return { exchange, dmi_energy, anisotropy, zeeman };
        }

        MicromagneticEnergyTerms energy_terms(const O3Field& field) const {
            const auto& lattice = field.lattice();
            const double area = lattice.dx() * lattice.dy();
            MicromagneticEnergyTerms result;
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    const auto terms = energy_density_terms_at(field, i, j);
                    result.exchange += area * terms.exchange;
                    result.dmi += area * terms.dmi;
                    result.anisotropy += area * terms.anisotropy;
                    result.zeeman += area * terms.zeeman;
                }
            }
            return result;
        }

        double energy(const O3Field& field) const override {
            return energy_terms(field).total();
        }

        Vec3 effective_field_at(
            const O3Field& field,
            std::size_t i,
            std::size_t j
        ) const {
            const Vec3 m = field(i, j);
            Vec3 result = 2.0 * exchange_
                * differential::laplacian(field, i, j);

            if (dmi_type_ == DMIType::Bulk) {
                result -= 2.0 * dmi_ * differential::curl(field, i, j);
            }
            else if (dmi_type_ == DMIType::Interfacial) {
                const Vec3 dx = differential::derivative_x(field, i, j);
                const Vec3 dy = differential::derivative_y(field, i, j);
                result += 2.0 * dmi_ * Vec3{
                    dx.z,
                    dy.z,
                    -dx.x - dy.y
                };
            }

            result += 2.0 * anisotropy_ * dot(m, easy_axis_) * easy_axis_;
            result += applied_field_;
            return result;
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
                            effective_field_at(field, i, j);
                    }
                }
            }
            return result;
        }

    private:
        double exchange_{};
        double dmi_{};
        double anisotropy_{};
        Vec3 applied_field_{};
        Vec3 easy_axis_{};
        DMIType dmi_type_{ DMIType::Bulk };
    };

} // namespace solitonkit
