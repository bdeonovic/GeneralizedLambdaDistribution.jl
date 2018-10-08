function m3(k, x)
  3*(-1)^k*(12*x^5+10*x^4-4*x^3-x^2+4*x+1)/(x^2*(x+1)*(2*x+1)*(3*x+1)) - 6*
  (-1)^k*(-digamma(1) + digamma(x+2))/(x^2*(x+1)) + 3*
  (-1)^k*(-digamma(1) + digamma(2*x+2))/(x^2*(2*x+1)) + 3*
  (-1)^k*(pi^2/6+(-digamma(1)+digamma(x+2))^2 - polygamma(1,x+2))/
  (x*(x+1))
end

function m4(x)
  4 * (144*x^7+156*x^6-18*x^5-24*x^4+7*x^3-11*x^2-7*x-1)/
  (x^3*(x+1)*(2*x+1)*(3*x+1)*(4*x+1)) +
  12 * (-digamma(1) + digamma(x+2))/(x^3*(x+1)) - 
  12 * (-digamma(1) + digamma(2*x+2))/(x^3*(2*x+1)) +
  4  * (-digamma(1) + digamma(3*x+2))/(x^3*(3*x+1)) -
  12 * (pi^2/6 + (-digamma(1) + digamma(x+2))^2 - polygamma(1,x+2))/(x^2*(x+1))+
  6  * (pi^2/6 + (-digamma(1) + digamma(2*x+2))^2 - polygamma(1,2*x+2))/
  (x^2*(2*x+1)) + 
  4 * (3 * (pi^2/6 - polygamma(1,x+2))*(-digamma(1) + digamma(x+2)) +  
       (-digamma(1) + digamma(x+2))^3 + 2*zeta(3) + polygamma(2,x+2) ) / (x*(x+1))
end

function vk(x1, x2)
  v = zeros(typeof(x1), 4)
  if x1 == 0
    if x2 == 0
      v[1] = 0.0
      v[2] = (pi ^ 2) / 3
      v[3] = 0.0
      v[4] = 7.0 * (pi ^ 4) / 15
    else
      v[1] = -x2 / (x2 + 1)
      e1 = 2 * (2 * x2 ^ 3 + x2 ^2 - x2 - 1) / 
      (x2 * (x2 + 1) * (2 * x2 + 1))
      e2 = 2 * (-digamma(1) + digamma(x2 + 2)) / (x2 * (x2 + 1))
      v[2] = e1 + e2
      v[3] = m3(3, x2)
      v[4] = m4(x2)
    end
  elseif x2 == 0
    v[1] = x1 / (x1+1)
    v[2] = 2*(2*x1^3+x1^2-x1-1)/(x1*(x1+1)*(2*x1+1)) +
    2*(-digamma(1) + digamma(x1+2))/(x1*(x1+1))
    v[3] = m3(4, x1)
    v[4] = m4(x1)
  else ## x1 != 0 && x2 != 0
    for k in 1:4
      s = 1.0
      r = 0.0
      for j in 0:k
        #b = beta(x1 * (k-j) + 1, x2 * j + 1)
        a = x1 * (k-j) + 1
        b = x2 * j + 1
        c = gamma(a) * gamma(b) / gamma(a+b)
        r += s * binomial(k,j) * c / (x1 ^ (k-j) * x2 ^ j)
        s = -s
      end
      v[k] = r
    end
  end
  return v
end

function fitgld(mu1::T, mu2::T, gamma1::T, gamma2::T, use_lbfgs=false) where T <: Real
  function f(x1, x2)
    v = vk(x1, x2)
    sigmal2sq = v[2] - v[1] ^ 2

    a3 = (v[3] - 3 * v[1] * v[2] + 2 * v[1] ^ 3) / (sigmal2sq ^ 1.5)
    a4 = (v[4] - 4 * v[1] * v[3] + 6 * v[1] ^ 2 * v[2] - 3 * v[1] ^ 4) / 
      (sigmal2sq ^ 2)

    return (gamma1 - a3) ^ 2 + (gamma2 - a4) ^ 2
  end
  m = use_lbfgs ? Model(solver=NLoptSolver(algorithm=:LD_LBFGS)) : 
    Model(solver=NLoptSolver(algorithm=:LN_NELDERMEAD))
  JuMP.register(m, :f, 2, f, autodiff=true)
  @variable(m, nlopt_lambda3, start = 0.01)
  @variable(m, nlopt_lambda4, start = 0.01)
  setlowerbound(nlopt_lambda3, -0.25 + eps())
  setlowerbound(nlopt_lambda4, -0.25 + eps())
  @NLobjective(m, Min, f(nlopt_lambda3, nlopt_lambda4))
  solve(m, suppress_warnings=true)

  lambda3 = getvalue(nlopt_lambda3)
  lambda4 = getvalue(nlopt_lambda4)
  v = vk(lambda3, lambda4)
  sigmal2sq = v[2] - v[1] ^2
  lambda2 = sqrt(sigmal2sq) / sqrt(mu2)
  lambda1 = 0.0
  if lambda3 == 0 || lambda4 == 0 || lambda3 == lambda4
    lambda1 = mu1 - v[1] / lambda2
  else
    lambda1 = mu1 - (v[1] - 1 / lambda3 + 1 / lambda4) / lambda2
  end
  return GLD(lambda1, lambda2, lambda3, lambda4)
end

