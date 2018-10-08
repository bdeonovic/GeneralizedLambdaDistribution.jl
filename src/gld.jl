struct GLD{T<:Real} <: ContinuousUnivariateDistribution
  lambda1::T
  lambda2::T
  lambda3::T
  lambda4::T

  function GLD{T}(lambda1::T, lambda2::T, lambda3::T, lambda4::T) where T <: Real
    @check_args(GLD, lambda3 > -0.25 && lambda4 > -0.25)
    new{T}(lambda1, lambda2, lambda3, lambda4)
  end
end

GLD(lambda1::T, lambda2::T, lambda3::T, lambda4::T) where T <: Real = 
  GLD{T}(lambda1, lambda2, lambda3, lambda4)
GLD(lambda1::Real, lambda2::Real, lambda3::Real, lambda4::Real) = 
  GLD(promote(lambda1, lambda2, lambda3, lambda4)...)
function GLD(lambda1::Integer, lambda2::Integer, lambda3::Integer, lambda4::Integer)
  GLD(Float64(lambda1), Float64(lambda2), Float64(lambda3), Float64(lambda4))
end

function minimum(d::GLD)
  if d.lambda3 > 0
    return d.lambda1 - 1 / (d.lambda2 * d.lambda3)
  else
    return -Inf
  end
end

function maximum(d::GLD)
  if d.lambda4 > 0
    return d.lambda1 + 1 / (d.lambda2 * d.lambda4)
  else
    return Inf
  end
end

#### Conversions
function convert(::Type{GLD{T}}, lambda1::Real, 
                 lambda2::Real, lambda3::Real, lambda4::Real) where T <: Real
  GLD(T(lambda1), T(lambda2), T(lambda3), T(lambda4))
end
function convert(::Type{GLD{T}}, d::GLD{S}) where {T <: Real, S <: Real}
  GLD(T(d.lambda1), T(d.lambda2), T(d.lambda3), T(d.lambda4))
end

#### Parameters

params(d::GLD) = (d.lambda1, d.lambda2, d.lambda3, d.lambda4)
@inline partype(d::GLD{T}) where T <: Real = T

#### Functions

function pdf(d::GLD, x::Real)
  if insupport(d, x)
    f = (u) -> quantile(d, u) - x
    u = fzero(f, 0.0, 1.0)
    return d.lambda2 / ( u ^ (d.lambda3 - 1) + (1 - u) ^ (d.lambda4 -1))
  else
    return 0.0
  end
end

function quantile(d::GLD, u::Real)
  if u > 1.0
    return 1.0
  elseif u < 0.0
    return 0.0
  else
    if d.lambda3 == 0
      if d.lambda4 == 0
        return d.lambda1 + 1 / d.lambda2 * log( u / (1 - u))
      elseif 0 < abs(d.lambda4) < Inf
        return d.lambda1 + 1 / d.lambda2 * (log(u) - ((1 - u) ^ d.lambda4 - 1) / d.lambda4)
      else
        return d.lambda1 + 1 / d.lambda2 * log(u)
      end
    elseif d.lambda4 == 0
      if 0 < abs(d.lambda3) < Inf
        return d.lambda1 + 1 / d.lambda2 * ( (u ^ d.lambda3 - 1) / d.lambda3 - log(1 - u))
      else
        return d.lambda1 - 1 / d.lambda2 * log(1-u)
      end
    else
      if d.lambda3 == Inf && (0 < d.lambda4 < Inf)
        return d.lambda1 - 1 / d.lambda2 * ( ((1 - u) ^ d.lambda4 - 1) / d.lambda4)
      else
        return d.lambda1 + 1 / d.lambda2 * ((u ^ d.lambda3 - 1) / d.lambda3 - 
                                      ((1-u) ^ d.lambda4 - 1) / d.lambda4)
      end
    end
  end
end
