#pragma once

#include <cmath>
#include <stdexcept>

namespace solitonkit {

    struct Vec3 {
        double x{};
        double y{};
        double z{};

        Vec3() = default;

        Vec3(double x_, double y_, double z_)
            : x(x_), y(y_), z(z_) {}

        Vec3 operator+(const Vec3& other) const {
            return { x + other.x, y + other.y, z + other.z };
        }

        Vec3 operator-(const Vec3& other) const {
            return { x - other.x, y - other.y, z - other.z };
        }

        Vec3 operator*(double a) const {
            return { a * x, a * y, a * z };
        }

        Vec3 operator/(double a) const {
            if (a == 0.0) {
                throw std::runtime_error("Division by zero in Vec3");
            }
            return { x / a, y / a, z / a };
        }

        Vec3& operator+=(const Vec3& other) {
            x += other.x;
            y += other.y;
            z += other.z;
            return *this;
        }

        Vec3& operator-=(const Vec3& other) {
            x -= other.x;
            y -= other.y;
            z -= other.z;
            return *this;
        }

        Vec3& operator*=(double a) {
            x *= a;
            y *= a;
            z *= a;
            return *this;
        }

        double norm_squared() const {
            return x * x + y * y + z * z;
        }

        double norm() const {
            return std::sqrt(norm_squared());
        }

        Vec3 normalized() const {
            const double n = norm();

            if (n == 0.0) {
                throw std::runtime_error("Cannot normalize zero Vec3");
            }

            return *this / n;
        }
    };

    inline Vec3 operator*(double a, const Vec3& v) {
        return v * a;
    }

    inline double dot(const Vec3& a, const Vec3& b) {
        return a.x * b.x + a.y * b.y + a.z * b.z;
    }

    inline Vec3 cross(const Vec3& a, const Vec3& b) {
        return {
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        };
    }

} // namespace solitonkit