// Experimental C++ parity/performance probe for the deterministic coordinate-polish kernel.
// This is not part of the public API and intentionally has no Python binding yet.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

struct Result {
    double fun{};
    int evaluations{};
    double seconds{};
};

static double ackley(const std::vector<double>& x) {
    const double n = static_cast<double>(x.size());
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (double v : x) {
        sum_sq += v * v;
        sum_cos += std::cos(2.0 * M_PI * v);
    }
    return -20.0 * std::exp(-0.2 * std::sqrt(sum_sq / n))
           - std::exp(sum_cos / n) + 20.0 + std::exp(1.0);
}

static std::vector<double> initial_point(int dimension, double lo, double hi, int seed) {
    std::vector<double> x(static_cast<std::size_t>(dimension));
    const double span = hi - lo;
    for (int i = 0; i < dimension; ++i) {
        const int q = (i * 37 + seed * 101 + 17) % 1000;
        x[static_cast<std::size_t>(i)] = lo + span * (static_cast<double>(q) / 999.0);
    }
    return x;
}

static Result polish_once(int dimension, int budget, int seed) {
    constexpr double lo = -32.768;
    constexpr double hi = 32.768;
    constexpr double step_fraction = 0.000625;
    const double span = hi - lo;
    std::vector<double> best_x = initial_point(dimension, lo, hi, seed);
    double best_f = ackley(best_x);
    int used = 1;
    double step = step_fraction * span;

    while (used + 2 * dimension <= budget && step > 1e-13 * span) {
        bool improved = false;
        for (int axis = 0; axis < dimension; ++axis) {
            for (double sign : {-1.0, 1.0}) {
                std::vector<double> cand = best_x;
                const std::size_t a = static_cast<std::size_t>(axis);
                cand[a] = std::clamp(cand[a] + sign * step, lo, hi);
                const double val = ackley(cand);
                ++used;
                if (val < best_f) {
                    best_x.swap(cand);
                    best_f = val;
                    improved = true;
                }
            }
        }
        if (!improved) step *= 0.25;
    }
    return {best_f, used, 0.0};
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: coordinate_polish_bench DIM BUDGET SEED REPEATS\n";
        return 2;
    }
    const int dimension = std::stoi(argv[1]);
    const int budget = std::stoi(argv[2]);
    const int seed = std::stoi(argv[3]);
    const int repeats = std::stoi(argv[4]);
    if (dimension < 2 || budget < 1 + 2 * dimension || repeats < 1) return 2;

    Result last{};
    const auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeats; ++r) last = polish_once(dimension, budget, seed);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();

    std::cout << std::setprecision(17)
              << "fun=" << last.fun
              << " evaluations=" << last.evaluations
              << " seconds=" << seconds
              << " repeats=" << repeats << "\n";
    return 0;
}
