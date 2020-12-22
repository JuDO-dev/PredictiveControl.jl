
abstract type PredictiveControlProblem end

struct PrimalNoCondensing <: PredictiveControlProblem end

struct PrimalFullCondensing <: PredictiveControlProblem end

struct DualFullCondensing <: PredictiveControlProblem end
