#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "solitonkit/core/Lattice3D.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    class O3Field3D {
    public:
        explicit O3Field3D(
            const Lattice3D& lattice,
            const Vec3& value = Vec3{ 0.0, 0.0, 1.0 },
            const Vec3& dirichlet_value = Vec3{ 0.0, 0.0, 1.0 }
        ) : lattice_(lattice),
            values_(lattice.size(), value.normalized()),
            dirichlet_value_(dirichlet_value.normalized())
        {
            enforce_boundary_condition();
        }

        const Lattice3D& lattice() const { return lattice_; }
        std::size_t size() const { return values_.size(); }

        Vec3& operator()(std::size_t i, std::size_t j, std::size_t k) {
            return values_[lattice_.index(i, j, k)];
        }

        const Vec3& operator()(
            std::size_t i,
            std::size_t j,
            std::size_t k
        ) const {
            return values_[lattice_.index(i, j, k)];
        }

        Vec3& at_index(std::size_t index) { return values_.at(index); }
        const Vec3& at_index(std::size_t index) const {
            return values_.at(index);
        }

        const Vec3& dirichlet_value() const { return dirichlet_value_; }

        void set_dirichlet_value(const Vec3& value) {
            dirichlet_value_ = value.normalized();
            enforce_boundary_condition();
        }

        void normalize_all() {
            for (auto& value : values_) {
                value = value.normalized();
            }
            enforce_boundary_condition();
        }

        void set_uniform(const Vec3& value) {
            const Vec3 normalized = value.normalized();
            for (auto& item : values_) {
                item = normalized;
            }
            enforce_boundary_condition();
        }

        void enforce_boundary_condition() {
            for (std::size_t k = 0; k < lattice_.nz(); ++k) {
                for (std::size_t j = 0; j < lattice_.ny(); ++j) {
                    for (std::size_t i = 0; i < lattice_.nx(); ++i) {
                        if (lattice_.is_dirichlet_boundary(i, j, k)) {
                            (*this)(i, j, k) = dirichlet_value_;
                        }
                    }
                }
            }
        }

        static O3Field3D random(
            const Lattice3D& lattice,
            unsigned int seed = 12345
        ) {
            O3Field3D field(lattice);
            std::mt19937 rng(seed);
            std::normal_distribution<double> distribution(0.0, 1.0);

            for (std::size_t index = 0; index < field.size(); ++index) {
                Vec3 value;
                do {
                    value = {
                        distribution(rng),
                        distribution(rng),
                        distribution(rng)
                    };
                } while (value.norm_squared() == 0.0);
                field.at_index(index) = value.normalized();
            }
            field.enforce_boundary_condition();
            return field;
        }

    private:
        Lattice3D lattice_;
        std::vector<Vec3> values_;
        Vec3 dirichlet_value_;
    };

} // namespace solitonkit
