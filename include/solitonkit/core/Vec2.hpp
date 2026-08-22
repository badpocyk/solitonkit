#pragma once

#include <cmath>
#include <stdexcept>

namespace solitonkit {

    struct Vec2 {
        double x{};
        double y{};

        Vec2() = default;
        Vec2(double x_, double y_) : x(x_), y(y_) {}

        Vec2 operator+(const Vec2& other) const {
            return { x + other.x, y + other.y };
        }

        Vec2 operator-(const Vec2& other) const {
            return { x - other.x, y - other.y };
        }

        Vec2 operator*(double scale) const {
            return { scale * x, scale * y };
        }

        Vec2 operator/(double scale) const {
            if (scale == 0.0) {
                throw std::runtime_error("Division by zero in Vec2");
            }
            return { x / scale, y / scale };
        }

        double norm_squared() const {
            return x * x + y * y;
        }

        double norm() const {
            return std::sqrt(norm_squared());
        }
    };

    inline Vec2 operator*(double scale, const Vec2& value) {
        return value * scale;
    }

    inline double dot(const Vec2& a, const Vec2& b) {
        return a.x * b.x + a.y * b.y;
    }

} // namespace solitonkit
