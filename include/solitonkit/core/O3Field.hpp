#pragma once

#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

#include "solitonkit/core/Lattice2D.hpp"
#include "solitonkit/core/Vec3.hpp"

namespace solitonkit {

    class O3Field {
    public:
        explicit O3Field(const Lattice2D& lattice)
            : lattice_(lattice),
            values_(lattice.size(), Vec3{ 0.0, 0.0, 1.0 })
        {}

        O3Field(const Lattice2D& lattice, const Vec3& value)
            : lattice_(lattice),
            values_(lattice.size(), value.normalized())
        {}

        const Lattice2D& lattice() const {
            return lattice_;
        }

        std::size_t size() const {
            return values_.size();
        }

        Vec3& operator()(std::size_t i, std::size_t j) {
            return values_[lattice_.index(i, j)];
        }

        const Vec3& operator()(std::size_t i, std::size_t j) const {
            return values_[lattice_.index(i, j)];
        }

        Vec3& at_index(std::size_t k) {
            return values_.at(k);
        }

        const Vec3& at_index(std::size_t k) const {
            return values_.at(k);
        }

        void set_uniform(const Vec3& value) {
            const Vec3 normalized_value = value.normalized();

            for (auto& v : values_) {
                v = normalized_value;
            }
        }

        void normalize_all() {
            for (auto& v : values_) {
                v = v.normalized();
            }
        }

        static O3Field uniform(const Lattice2D& lattice, const Vec3& value) {
            return O3Field(lattice, value);
        }

        static O3Field random(const Lattice2D& lattice, unsigned int seed = 12345) {
            O3Field field(lattice);

            std::mt19937 rng(seed);
            std::normal_distribution<double> dist(0.0, 1.0);

            for (std::size_t k = 0; k < field.size(); ++k) {
                Vec3 v{ dist(rng), dist(rng), dist(rng) };

                while (v.norm_squared() == 0.0) {
                    v = Vec3{ dist(rng), dist(rng), dist(rng) };
                }

                field.at_index(k) = v.normalized();
            }

            return field;
        }

    private:
        Lattice2D lattice_;
        std::vector<Vec3> values_;
    };

} // namespace solitonkit