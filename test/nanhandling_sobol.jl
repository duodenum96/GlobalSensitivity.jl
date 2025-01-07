# Tests for nan handling
# Make a differential equation problem that can be divergent. 

using Revise, Test, OrdinaryDiffEq, GlobalSensitivity, Plots, QuasiMonteCarlo, Statistics

function f(du, u, p, t)
    du .= p[1] * u
    return du
end

u0 = [1.0]
p = [50] # diverges
tspan = (0.0, 20.0)
t = collect(range(0, stop = 10, length = 200))
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob, Tsit5(); saveat = t)

f1 = let prob = prob, t = t
    function (p)
        prob1 = remake(prob; p = p)
        sol = solve(prob1, Tsit5(); saveat = t)
        if sol.retcode == ReturnCode.Success
            return mean(Array(sol))
        else
            return NaN
        end
    end
end

samples = 100
lb = [20]
ub = [70]
sampler = SobolSample()
A, B = QuasiMonteCarlo.generate_design_matrices(samples, lb, ub, sampler)

m = gsa(f1, Sobol(), A, B; dropnan = true)
@test !any(isnan.(m.S1))
@test !any(isnan.(m.ST))
@test !any(isnan.(m.S1_Conf_Int))
@test !any(isnan.(m.ST_Conf_Int))
