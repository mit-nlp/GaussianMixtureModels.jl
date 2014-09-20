# GaussianMixtureModels.jl
#
# author: Wade Shen
# swade@ll.mit.edu
# Copyright © 2009 Massachusetts Institute of Technology, Lincoln Laboratory
module GaussianMixtureModels
export Gaussian, GMM, E, M, train, par_train, split, logsum, total_ll, ll, DiagGaussian, dims, order
using Stage
import Base: show, zero

# -------------------------------------------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------------------------------------------
const Log2π = log(2.0 * π)

# -------------------------------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------------------------------
function logsum(values :: Vector)
  total = 0.0
  maxval = maximum(values)
  for val in values
    total += exp(val - maxval)
  end

  return log(total) + maxval
end
logsum(values...) = logsum([values...])

# -------------------------------------------------------------------------------------------------------------------------
# Types
# -------------------------------------------------------------------------------------------------------------------------
abstract Gaussian
type DiagGaussian <: Gaussian
  μ        :: Vector{Float32}
  σ        :: Diagonal{Float32}
  logdet_σ :: Float32
  inv_σ    :: Diagonal{Float32}
end

DiagGaussian(dims :: Int) = DiagGaussian(zeros(dims), Diagonal(ones(dims)), log(1.0), inv(Diagonal(ones(dims))))
  
type GMM
  mix        :: Vector{Gaussian}
  logweights :: Vector{Float32}
  workspace  :: Vector{Float32}
end

GMM(dims :: Int, order :: Int) =  GMM([ DiagGaussian(dims) for i = 1:order ], log([ 1.0 / float32(order) for i = 1:order]), zeros(order))

type Acc
  zero           :: Vector{Float32}
  one            :: Vector{Vector{Float32}}
  two            :: Vector{Vector{Float32}}
  N              :: Float32
  log_likelihood :: Float32
end

Acc(order :: Int, dims :: Int) = Acc(zeros(order), [ zeros(dims) for i = 1:order ], [ zeros(dims) for i = 1:order ], 0.0, 0.0)

# -------------------------------------------------------------------------------------------------------------------------
# Methods
# -------------------------------------------------------------------------------------------------------------------------
zero(a::Acc) = Acc(order(a), dims(a))
function (+)(a1 :: Acc, a2 :: Acc) 
  sumone = Array(Vector{Float32}, length(a1.one))
  sumtwo = Array(Vector{Float32}, length(a1.two))
  for i = 1:length(a1.one)
    sumone[i] = a1.one[i] + a2.one[i]
    sumtwo[i] = a1.two[i] + a2.two[i]
  end

  return Acc(a1.zero + a2.zero, sumone, sumtwo, a1.N + a2.N, a1.log_likelihood + a2.log_likelihood)
end
dims(g :: Gaussian) = length(g.μ)
dims(gmm :: GMM)    = dims(gmm.mix[1])
dims(acc :: Acc)    = length(acc.one[1])
order(gmm :: GMM)   = length(gmm.mix)
order(acc :: Acc)   = length(acc.zero)

function ll(g :: Gaussian, x :: Vector{Float32})
  xm = x - g.μ
  return -0.5 * (dims(g) * Log2π + g.logdet_σ + (xm' * g.inv_σ * xm)[1])
end

function ll(gmm :: GMM, x :: Vector{Float32})
  for i = 1:order(gmm)
    gmm.workspace[i] = gmm.logweights[i] + ll(gmm.mix[i], x)
  end

  return logsum(gmm.workspace)
end
  
function total_ll(gmm :: GMM, data) # iterable over data
  total = 0.0
  for d in data
    total += ll(gmm, d)
  end

  return total
end

function E(gmm :: GMM, data)
  acc = Acc(order(gmm), dims(gmm))

  for d in data
    total = ll(gmm, d)
    for i = 1:order(gmm)
      post = exp(gmm.workspace[i] - total)
      acc.zero[i] += post
      acc.one[i]  += post * d
      acc.two[i]  += post * abs2(d) #- gmm.mix[i].μ dot(d - gmm.mix[i].μ, d - gmm.mix[i].μ)
    end
    acc.N              += 1.0
    acc.log_likelihood += total
  end

  return acc
end

function M(gmm :: GMM, acc :: Acc, floor_w = 1e-4, floor_σ = 1e-3)
  for i = 1:order(gmm)
    w                 = acc.zero[i] / acc.N
    gmm.logweights[i] = log(w < floor_w ? floor_w : w)
    gmm.mix[i].μ      = acc.one[i] / acc.zero[i]
    σ                 = (acc.two[i] / acc.zero[i]) - abs2(gmm.mix[i].μ)
    for d = 1:dims(gmm)
      if σ[d] < floor_σ
        σ[d] = floor_σ
      end
    end
    gmm.mix[i].σ        = Diagonal(σ)
    gmm.mix[i].logdet_σ = sum(log(σ)) # this could underflow: log(det(gmm.mix[i].σ))
    gmm.mix[i].inv_σ    = inv(gmm.mix[i].σ)
  end

  return gmm
end

function train(gmm :: GMM, data; iterations = 5)
  for i = 1:iterations
    acc = E(gmm, data)
    @info @sprintf("iteration %3d, log likelihood: %10.3f (%7d exemplars, average per exemplar likelihood = %10.5f)", i, acc.log_likelihood, acc.N, acc.log_likelihood / acc.N)
    M(gmm, acc)
  end

  return gmm
end

# assumes that data is a vector of iterables
function par_train(gmm :: GMM, data :: Vector; iterations = 5)
  ret = gmm
  for i = 1:iterations
    acc = @parallel (+) for sf in data
      E(ret, sf)
    end
    @info @sprintf("iteration %3d, log likelihood: %10.3f (%7d exemplars, average per exemplar likelihood = %10.5f)", i, acc.log_likelihood, acc.N, acc.log_likelihood / acc.N)
    ret = M(gmm, acc)
    end
end

function preturb(g :: DiagGaussian, delta)
  return DiagGaussian(g.μ + (delta * diag(g.σ)), g.σ, g.logdet_σ, g.inv_σ)
end

function split(gmm :: GMM, delta = 0.2)
  workspace  = zeros(order(gmm) * 2)
  logweights = Array(Float32, order(gmm) * 2)
  mix        = Array(Gaussian, order(gmm) * 2)
  for i = 1:order(gmm)
    logweights[i] = gmm.logweights[i] - log(2.0)
    mix[i]        = preturb(gmm.mix[i], delta)
  end
  for i = 1:order(gmm)
    logweights[i + order(gmm)] = gmm.logweights[i] - log(2.0)
    mix[i + order(gmm)]        = preturb(gmm.mix[i], -delta)
  end

  return GMM(mix, logweights, workspace)
end

end # module
