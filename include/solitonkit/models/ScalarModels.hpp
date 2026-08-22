#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "solitonkit/core/ScalarField2D.hpp"
#include "solitonkit/models/Model.hpp"
#include "solitonkit/operators/DifferentialOperators.hpp"

namespace solitonkit {

    class Phi4Model : public DifferentiableModel<ScalarField2D, double> {
    public:
        Phi4Model(double lambda = 1.0, double vacuum = 1.0)
            : lambda_(lambda), vacuum_(vacuum)
        {
            if (lambda_ <= 0.0 || vacuum_ <= 0.0) {
                throw std::runtime_error(
                    "Phi4Model lambda and vacuum must be positive"
                );
            }
        }

        std::string name() const override { return "phi4"; }
        std::size_t dimensions() const override { return 2; }
        FieldKind field_kind() const override { return FieldKind::Scalar2D; }
        double lambda() const { return lambda_; }
        double vacuum() const { return vacuum_; }

        double energy_density_at(
            const ScalarField2D& field,
            std::size_t i,
            std::size_t j
        ) const {
            const double dx = differential::derivative_x(field, i, j);
            const double dy = differential::derivative_y(field, i, j);
            const double phi = field(i, j);
            const double well = phi * phi - vacuum_ * vacuum_;
            return 0.5 * (dx * dx + dy * dy)
                + 0.25 * lambda_ * well * well;
        }

        double energy(const ScalarField2D& field) const override {
            const auto& lattice = field.lattice();
            const double area = lattice.dx() * lattice.dy();
            double result = 0.0;
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    result += area * energy_density_at(field, i, j);
                }
            }
            return result;
        }

        std::vector<double> negative_gradient(
            const ScalarField2D& field
        ) const override {
            const auto& lattice = field.lattice();
            std::vector<double> result(field.size(), 0.0);
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    if (lattice.is_fixed_boundary(i, j)) {
                        continue;
                    }
                    const double phi = field(i, j);
                    result[lattice.index(i, j)] =
                        differential::laplacian(field, i, j)
                        - lambda_ * phi * (phi * phi - vacuum_ * vacuum_);
                }
            }
            return result;
        }

    private:
        double lambda_{};
        double vacuum_{};
    };

    class SineGordonModel
        : public DifferentiableModel<ScalarField2D, double> {
    public:
        SineGordonModel(double mass = 1.0, double beta = 1.0)
            : mass_(mass), beta_(beta)
        {
            if (mass_ <= 0.0 || beta_ <= 0.0) {
                throw std::runtime_error(
                    "SineGordonModel mass and beta must be positive"
                );
            }
        }

        std::string name() const override { return "sine-gordon"; }
        std::size_t dimensions() const override { return 2; }
        FieldKind field_kind() const override { return FieldKind::Scalar2D; }
        double mass() const { return mass_; }
        double beta() const { return beta_; }

        double energy_density_at(
            const ScalarField2D& field,
            std::size_t i,
            std::size_t j
        ) const {
            const double dx = differential::derivative_x(field, i, j);
            const double dy = differential::derivative_y(field, i, j);
            const double potential = mass_ * mass_
                * (1.0 - std::cos(beta_ * field(i, j)))
                / (beta_ * beta_);
            return 0.5 * (dx * dx + dy * dy) + potential;
        }

        double energy(const ScalarField2D& field) const override {
            const auto& lattice = field.lattice();
            const double area = lattice.dx() * lattice.dy();
            double result = 0.0;
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    result += area * energy_density_at(field, i, j);
                }
            }
            return result;
        }

        std::vector<double> negative_gradient(
            const ScalarField2D& field
        ) const override {
            const auto& lattice = field.lattice();
            std::vector<double> result(field.size(), 0.0);
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    if (lattice.is_fixed_boundary(i, j)) {
                        continue;
                    }
                    result[lattice.index(i, j)] =
                        differential::laplacian(field, i, j)
                        - (mass_ * mass_ / beta_)
                            * std::sin(beta_ * field(i, j));
                }
            }
            return result;
        }

    private:
        double mass_{};
        double beta_{};
    };

    class XYModel : public DifferentiableModel<XYField, double> {
    public:
        XYModel(double coupling = 1.0, double field_strength = 0.0)
            : coupling_(coupling), field_strength_(field_strength)
        {
            if (coupling_ <= 0.0 || field_strength_ < 0.0) {
                throw std::runtime_error(
                    "XYModel coupling must be positive and field non-negative"
                );
            }
        }

        std::string name() const override { return "xy"; }
        std::size_t dimensions() const override { return 2; }
        FieldKind field_kind() const override { return FieldKind::XY2D; }
        double coupling() const { return coupling_; }
        double field_strength() const { return field_strength_; }

        double energy(const XYField& field) const override {
            const auto& lattice = field.lattice();
            const double area = lattice.dx() * lattice.dy();
            double result = 0.0;
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    const double theta = field(i, j);
                    const double x_bond = (i + 1 < lattice.nx()
                            || lattice.boundary_x() == BoundaryCondition::Periodic)
                        ? coupling_ * (1.0 - std::cos(
                            field(lattice.right(i), j) - theta
                        )) / (lattice.dx() * lattice.dx())
                        : 0.0;
                    const double y_bond = (j + 1 < lattice.ny()
                            || lattice.boundary_y() == BoundaryCondition::Periodic)
                        ? coupling_ * (1.0 - std::cos(
                            field(i, lattice.up(j)) - theta
                        )) / (lattice.dy() * lattice.dy())
                        : 0.0;
                    result += area * (
                        x_bond + y_bond
                        + field_strength_ * (1.0 - std::cos(theta))
                    );
                }
            }
            return result;
        }

        std::vector<double> negative_gradient(
            const XYField& field
        ) const override {
            const auto& lattice = field.lattice();
            std::vector<double> result(field.size(), 0.0);
            for (std::size_t j = 0; j < lattice.ny(); ++j) {
                for (std::size_t i = 0; i < lattice.nx(); ++i) {
                    if (lattice.is_fixed_boundary(i, j)) {
                        continue;
                    }
                    const double theta = field(i, j);
                    double force = -field_strength_ * std::sin(theta);
                    if (i > 0 || lattice.boundary_x() == BoundaryCondition::Periodic) {
                        force += coupling_ * std::sin(
                            field(lattice.left(i), j) - theta
                        ) / (lattice.dx() * lattice.dx());
                    }
                    if (i + 1 < lattice.nx()
                        || lattice.boundary_x() == BoundaryCondition::Periodic) {
                        force += coupling_ * std::sin(
                            field(lattice.right(i), j) - theta
                        ) / (lattice.dx() * lattice.dx());
                    }
                    if (j > 0 || lattice.boundary_y() == BoundaryCondition::Periodic) {
                        force += coupling_ * std::sin(
                            field(i, lattice.down(j)) - theta
                        ) / (lattice.dy() * lattice.dy());
                    }
                    if (j + 1 < lattice.ny()
                        || lattice.boundary_y() == BoundaryCondition::Periodic) {
                        force += coupling_ * std::sin(
                            field(i, lattice.up(j)) - theta
                        ) / (lattice.dy() * lattice.dy());
                    }
                    result[lattice.index(i, j)] = force;
                }
            }
            return result;
        }

    private:
        double coupling_{};
        double field_strength_{};
    };

} // namespace solitonkit
