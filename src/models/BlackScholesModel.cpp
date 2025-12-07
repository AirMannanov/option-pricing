#include <cmath>
#include <algorithm>

#include "../../include/pricing/models/BlackScholesModel.hpp"

namespace pricing {
namespace models {

core::PricingResult BlackScholesModel::price(
    const core::Option& option,
    const core::MarketData& marketData) const {

    double S = marketData.getSpot();
    double K = option.getStrike();
    double r = marketData.getRiskFreeRate();
    double sigma = marketData.getVolatility();
    double T = option.getTimeToExpiration();

    // Handle edge cases
    if (T == 0.0) {
        // At expiration, option value is intrinsic value
        core::PricingResult result;
        if (option.isCall()) {
            result.price = std::max(S - K, 0.0);
        } else {
            result.price = std::max(K - S, 0.0);
        }
        return result;
    }

    if (sigma == 0.0) {
        // No volatility: option value is discounted intrinsic value
        core::PricingResult result;
        double discountFactor = std::exp(-r * T);
        if (option.isCall()) {
            result.price = std::max(S - K * discountFactor, 0.0);
        } else {
            result.price = std::max(K * discountFactor - S, 0.0);
        }
        return result;
    }

    double d1 = calculateD1(S, K, r, sigma, T);
    double d2 = calculateD2(d1, sigma, T);

    core::PricingResult result;
    if (option.isCall()) {
        result.price = calculateCallPrice(S, K, r, T, d1, d2);
    } else {
        result.price = calculatePutPrice(S, K, r, T, d1, d2);
    }

    return result;
}

double BlackScholesModel::normalCDF(double x) {
    // Approximation of the cumulative distribution function
    // Using Abramowitz and Stegun approximation
    // https://labex.io/tutorials/c-compute-a-cumulative-distribution-function-cdf-in-c-435339
    const double a1 =  0.254829592;
    const double a2 = -0.284496736;
    const double a3 =  1.421413741;
    const double a4 = -1.453152027;
    const double a5 =  1.061405429;
    const double p  =  0.3275911;

    int sign = 1;
    if (x < 0) {
        sign = -1;
        x = -x;
    }

    double t = 1.0 / (1.0 + p * x);
    double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * std::exp(-x * x);

    return 0.5 * (1.0 + sign * y);
}

double BlackScholesModel::calculateD1(double S, double K, double r, double sigma, double T) {
    if (S <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    double numerator = std::log(S / K) + (r + 0.5 * sigma * sigma) * T;
    double denominator = sigma * std::sqrt(T);
    return numerator / denominator;
}

double BlackScholesModel::calculateD2(double d1, double sigma, double T) {
    return d1 - sigma * std::sqrt(T);
}

double BlackScholesModel::calculateCallPrice(double S, double K, double r, double T, double d1, double d2) {
    double N_d1 = normalCDF(d1);
    double N_d2 = normalCDF(d2);
    double discountFactor = std::exp(-r * T);
    return S * N_d1 - K * discountFactor * N_d2;
}

double BlackScholesModel::calculatePutPrice(double S, double K, double r, double T, double d1, double d2) {
    double N_minus_d1 = normalCDF(-d1);
    double N_minus_d2 = normalCDF(-d2);
    double discountFactor = std::exp(-r * T);
    return K * discountFactor * N_minus_d2 - S * N_minus_d1;
}

} // namespace models
} // namespace pricing
