using GaussianMixtureModels
using Base.Test
using Stage

# Test components
@expect logsum([-4.252368,-3.4738824]) == log(sum(exp.([-4.252368,-3.4738824])))
@expect logsum([-0.69314718055995, -0.69314718055995]) < 1e-10

# Fit random data
srand(0)
dims = 2
data = [ map(Float32, randn(dims)) for i = 1:1500 ]
@info "mean: $(mean(data))"
dv = zeros(dims)
for d in data
  dv += abs2.(d)
end
@info "diag(cov): $(dv/length(data) - abs2.(mean(data)))"

# Python tests
# using PyCall
# @pyimport sklearn.mixture as py
# sigma = Diagonal(float32(dv/length(data) - abs2(mean(data))))
# g = py.GMM(n_components = 1, n_iter = 1, init_params = "")
# g[:means_] = zeros(dims)'
# g[:covars_] = ones(dims)'
# @debug "init likelihood of pygmm = $(sum(g[:score](data)))"
# @debug "init likelihood of jlgmm = $(total_ll(gmm, data))"
# g[:fit](data)
# train(gmm, data, iterations = 1)
# @debug "1st likelihood of pygmm = $(sum(g[:score](data)))"
# @debug "\tpy mean  = $(g[:means_])"
# @debug "\tpy covar = $(g[:covars_])"
# gmm.mix[1].μ = vec(g[:means_])
# gmm.mix[1].σ = Diagonal(vec(g[:covars_]))
# gmm.mix[1].logdet_σ = 0.005992144035145126 * 2.0 # log(norm(gmm.mix[1].σ))
# gmm.mix[1].inv_σ  = inv(gmm.mix[1].σ)
# @debug "1st likelihood of jlgmm = $(total_ll(gmm, data))"
# @debug "\tjl mean  = $(gmm.mix[1].μ)"
# @debug "\tjl covar = $(gmm.mix[1].σ)"

gmm = GMM(dims, 1)
train(gmm, [ data ] , iterations = 10)
gmm2 = GaussianMixtureModels.split(gmm) # 2g
train(gmm2, [ data ], iterations = 10)
gmm3 = GaussianMixtureModels.split(gmm2) # 4g
train(gmm3, [ data ], iterations = 10)
println(gmm3)
