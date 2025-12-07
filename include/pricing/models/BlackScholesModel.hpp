#ifndef PRICING_MODELS_BLACK_SCHOLES_MODEL_HPP
#define PRICING_MODELS_BLACK_SCHOLES_MODEL_HPP

#include "PricingModel.hpp"

namespace pricing {
namespace models {

class BlackScholesModel : public PricingModel {
public:
    core::PricingResult price(
        const core::Option& option,
        const core::MarketData& marketData) const override;

private:
    static double normalCDF(double x);
    static double calculateD1(double S, double K, double r, double sigma, double T);
    static double calculateD2(double d1, double sigma, double T);
    static double calculateCallPrice(double S, double K, double r, double T, double d1, double d2);
    static double calculatePutPrice(double S, double K, double r, double T, double d1, double d2);
};

} // namespace models
} // namespace pricing

#endif // PRICING_MODELS_BLACK_SCHOLES_MODEL_HPP

