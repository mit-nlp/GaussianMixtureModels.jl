@everywhere using GaussianMixtureModels, Stage, Features, Iterators, Clustering, DocOpt
@everywhere import Base: start, next, done

# -------------------------------------------------------------------------------------------------------------------------
# Lazy Map
# -------------------------------------------------------------------------------------------------------------------------
@everywhere immutable LazyMap{I}
  flt::Function
  itr::I
end
@everywhere lazy_map(f::Function, itr) = LazyMap(f, itr)

@everywhere function start(m :: LazyMap) 
  s = start(m.itr)
  return s
end

@everywhere function next(m :: LazyMap, s) 
  n, ns = next(m.itr, s)
  return (m.flt(n), ns)
end

@everywhere done(m :: LazyMap, s) = done(m.itr, s)

# -------------------------------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------------------------------
function speech_frames(sf :: SegmentedFile)
  m, n = mask(sf)
  lazy_map(x -> x[2], filter(f -> m[f[1]], enumerate(HTKFeatures(sf.fn))))
end

function nonspeech_frames(sf :: SegmentedFile)
  m, n = mask(sf)
  lazy_map(x -> x[2], filter(f -> !m[f[1]], enumerate(HTKFeatures(sf.fn))))
end

@everywhere function kmeans_init(gmm :: GMM, data; sample_size = 1500, floor_σ = 1e-3)
  mat = Array(Float32, length(gmm.mix[1].μ), 0)
  @timer "kmeans clustering frames" begin
    @info "starting data sampling"
    for sp in data
      n = 0
      for f in sp
        if n < sample_size
          mat = hcat(mat, f)
        end
        n += 1
      end
    end
    @info "starting k-means"
    km = kmeans(mat, order(gmm))
  end

  totalw = sum(km.cweights)
  for i = 1:size(km.centers, 2)
    gmm.logweights[i] = log(km.cweights[i] / totalw)
    gmm.mix[i].μ      = vec(km.centers[:, i])
    ssum              = zeros(size(km.centers, 1))
    n                 = 0
    for d = 1:size(mat, 2)
      if km.assignments[d] == i
        ssum += abs2(vec(mat[:, d]))
        n += 1
      end
    end
    σ = (ssum / km.counts[i]) - abs2(gmm.mix[i].μ)
    for d = 1:size(km.centers, 1)
      if σ[d] < floor_σ
        σ[d] = floor_σ
      end
    end
    gmm.mix[i].σ        = Diagonal(σ)
    gmm.mix[i].logdet_σ = sum(log(σ)) #log(det(gmm.mix[i].σ))
    gmm.mix[i].inv_σ    = inv(gmm.mix[i].σ)
  end
  return gmm
end

# -------------------------------------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------------------------------------
function trn(ana, dir; splits = 6, iterations = 5)
  files     = analist(ana, dir = dir)
  speech    = map(speech_frames, files)
  nonspeech = map(nonspeech_frames, files)
  
  d             = dims(HTKFeatures(files[1].fn))
  speech_gmm    = GMM(d, 1)
  nonspeech_gmm = GMM(d, 1)
  for i = 1:splits
    @timer "training $(2^(i-1))g speech models" train(speech_gmm, chain(speech...), iterations = iterations)
    @timer "training $(2^(i-1))g non-speech models" train(nonspeech_gmm, chain(nonspeech...), iterations = iterations)
    speech_gmm    = split(speech_gmm)
    nonspeech_gmm = split(nonspeech_gmm)
  end
  @timer "final training of speech models" train(speech_gmm, chain(speech...), iterations = iterations)
  @timer "final training of non-speech models" train(nonspeech_gmm, chain(nonspeech...), iterations = iterations)

  return speech_gmm, nonspeech_gmm
end

function ktrain(ana, dir; g = 64, iterations = 5)
  files     = analist(ana, dir = dir)
  speech    = map(speech_frames, files)
  nonspeech = map(nonspeech_frames, files)
  
  d             = dims(HTKFeatures(files[1].fn))
  spmat         = Array(Float32, d, 0)
  nsmat         = Array(Float32, d, 0)

  speech_gmm    = @spawn kmeans_init(GMM(d, g), speech)
  nonspeech_gmm = @spawn kmeans_init(GMM(d, g), nonspeech)
  
  @timer "$(g)g training of speech models"     par_train(fetch(speech_gmm), speech, iterations = iterations)
  @timer "$(g)g training of non-speech models" par_train(fetch(nonspeech_gmm), nonspeech, iterations = iterations)

  return fetch(speech_gmm), fetch(nonspeech_gmm)
end

# -------------------------------------------------------------------------------------------------------------------------
# Score and optimize
# -------------------------------------------------------------------------------------------------------------------------
function score(files, speech, nonspeech; window_radius = 10)
  scores = Float32[]
  for sf in files
    for (i, f) in enumerate(HTKFeatures(sf.fn))
      scr = ll(speech, f) - ll(nonspeech, f)
      push!(scores, scr)
    end
  end

  for s = 1:length(scores)
    window    = scores[max(1, s-window_radius):min(end, s+window_radius)]
    ns        = mean(window)
    scores[s] = ns
  end
  
  return scores
end

function test(ana, dir, speech, nonspeech; threshold = 0.0)
  files  = analist(ana, dir = dir)
  scores = score(files, speech, nonspeech)
  N      = 0
  FAs    = 0
  misses = 0
  decs   = (String=>Vector{Bool})[]
  for sf in files
    speech_mask, speech_frames = mask(sf)
    decisions = [ false for i = 1:length(speech_mask) ]
    for i = 1:length(speech_mask)
      score = scores[N+1]
      if (score < threshold) && speech_mask[i]
        misses += 1
      elseif score >= threshold && !speech_mask[i]
        FAs += 1
      end
      if score >= threshold
        decisions[i] = true
      end
      N += 1
    end
    decs[sf.fn] = decisions
  end

  return FAs, misses, N, decs #[ x >= threshold for x in scores ]
end

function optimize(ana, dir, speech, nonspeech; c_miss = 1.0, c_fa = 1.0)
  files  = analist(ana, dir = dir)
  scores = score(files, speech, nonspeech)
  truth  = Bool[]

  N_speech = 0
  N        = 0
  N_ns     = 0
  for sf in files
    speech_mask, speech_frames = mask(sf)
    for i = 1:length(speech_mask)
      if speech_mask[i]
        push!(truth, true)
        N_speech += 1
      else
        push!(truth, false)
        N_ns += 1
      end
      N += 1
    end
  end

  dc   = zeros(length(scores))
  fa   = zeros(length(scores))
  miss = zeros(length(scores))
  hits = 0
  fas  = 0
  eer  = 0.0
  eerd = 10000.0

  indexes = sortperm(scores, rev = true)
  for idx in indexes
    scr     = scores[idx]
    speechp = truth[idx]
    if speechp
      hits += 1
    else
      fas += 1
    end

    far       = fas / float32(N_ns)
    missr     = (N_speech - hits) / float32(N_speech)
    fa[idx]   = far
    miss[idx] = missr
    dc[idx]   = far * c_fa + missr * c_miss

    if eerd > abs(fa[idx] - miss[idx])
      eerd = abs(fa[idx] - miss[idx])
      eer  = (fa[idx] + miss[idx]) / 2.0
    end
  end

  best = indmin(dc)
  return scores[best], fa[best], miss[best], dc[best], fa, miss, eer, N_speech, N_ns
end

# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

usage = """Analist SAD Train/test script
Usage:
  anasad.jl [options]

Options:
  --audio=dir, -a    Audio Directory relative to current path [default: rats-sample]
  --test=M           Analist for testing and threshold optimization
  --train=M          Analist with marks for training
  --kmeans           Do K-Means Initialization instead of binary splitting [default: false]
  --model=s          Name of output model [default: rats-sad.gmm]
  --iterations=i     Number of EM training iterations to perform during GMM training [default: 5]
  --gaussians=g, -g  Number of gaussians to target for final GMM (should be a power of two if binary splitting [default: 2]
  --out=O, -o        Write output analist
"""

args = docopt(usage, ARGS, version=v"0.0.1")

# train
if args["--train"] != nothing
  @info "training with $(args["--train"]) using audio from $(args["--audio"]) with $(args["--gaussians"]) gaussians and $(args["--iterations"]) iterations"
  speech, nonspeech = args["--kmeans"] ? ktrain(args["--train"], args["--audio"], g = int(args["--gaussians"]), iterations = int(args["--iterations"])) :
                                         trn(args["--train"], args["--audio"], splits = int(log2(int(args["--gaussians"]))), iterations = int(args["--iterations"]))
  outf = open(args["--model"], "w")
  serialize(outf, speech)
  serialize(outf, nonspeech)
  close(outf)
end

# score
if args["--test"] != nothing
  f = open(args["--model"], "r")
  speech    = deserialize(f)
  nonspeech = deserialize(f)
  close(f)

  threshold, opt_fa, opt_miss, opt_dc, fa, miss, eer, N_speech, N_ns = optimize(args["--test"], args["--audio"], speech, nonspeech)
  @info "optimum threshold = $threshold [eer = $eer, fa rate = $opt_fa, miss rate = $opt_miss, decision cost = $opt_dc]"

  FAs, misses, N, decisions = test(args["--test"], args["--audio"], speech, nonspeech, threshold = threshold)
  @info "FA rate   = $(FAs / N_ns) (N = $FAs / $N_ns)"
  @info "Miss rate = $(misses / N_speech) (N = $misses / $N_speech)"
  @info "N         = $N"
  if args["--out"] != nothing
    segid = 0
    outdir = args["--out"]
    isdir(outdir) || mkdir(outdir)
    for (k, decs) in decisions
      base, ext = splitext(basename(k))
      out = open("$outdir/$base.lab", "w")
      for i = 1:length(decs)
        start = decs[i] && (i == 1 || (i > 1 && decs[i - 1] == false))
        if start
          j      = i
          segid += 1
          while (j < length(decs)) && decs[j]
            j += 1
          end
          @printf(out, "%f %f speech\n", (i / 100.0), (j / 100.0))
          #@printf(out, "%s-%05d %10d %10d %s\n", basename(k), segid, round(i * 100.0 * 8000.0), round(j * 100.0 * 8000.0), basename(k))
        end
      end
      close(out)
    end
  end
end
