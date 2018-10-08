module GeneralizedLambdaDistribution
using Distributions
using Roots
using SpecialFunctions
using NLopt
using JuMP

import Base.convert
import Distributions: params, partype, pdf, logpdf, logcdf, quantile, rand, cdf, 
       @distr_support, @check_args, minimum, maximum

include("gld.jl")
include("fitgld.jl")

export
GLD

end # module
