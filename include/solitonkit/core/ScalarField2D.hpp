#pragma once

#include <cstddef>
#include <vector>

#include "solitonkit/core/Lattice2D.hpp"

namespace solitonkit {

    class ScalarField2D {
    public:
        explicit ScalarField2D(
            const Lattice2D& lattice,
            double value = 0.0,
            double dirichlet_value = 0.0
        ) : lattice_(lattice),
            values_(lattice.size(), value),
            dirichlet_value_(dirichlet_value)
        {
            enforce_boundary_condition();
        }

        const Lattice2D& lattice() const { return lattice_; }
        std::size_t size() const { return values_.size(); }

        double& operator()(std::size_t i, std::size_t j) {
            return values_[lattice_.index(i, j)];
        }

        const double& operator()(std::size_t i, std::size_t j) const {
            return values_[lattice_.index(i, j)];
        }

        double& at_index(std::size_t index) { return values_.at(index); }
        const double& at_index(std::size_t index) const {
            return values_.at(index);
        }

        double dirichlet_value() const { return dirichlet_value_; }

        void set_dirichlet_value(double value) {
            dirichlet_value_ = value;
            enforce_boundary_condition();
        }

        void set_uniform(double value) {
            for (auto& item : values_) {
                item = value;
            }
            enforce_boundary_condition();
        }

        void enforce_boundary_condition() {
            for (std::size_t j = 0; j < lattice_.ny(); ++j) {
                for (std::size_t i = 0; i < lattice_.nx(); ++i) {
                    if (lattice_.is_dirichlet_boundary(i, j)) {
                        (*this)(i, j) = dirichlet_value_;
                    }
                }
            }
        }

    private:
        Lattice2D lattice_;
        std::vector<double> values_;
        double dirichlet_value_{};
    };

    class XYField : public ScalarField2D {
    public:
        using ScalarField2D::ScalarField2D;
    };

} // namespace solitonkit
