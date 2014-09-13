using GaussianMixtureModel, Stage, Features, Iterators
import Base: start, next, done

immutable Map{I}
    flt::Function
    itr::I
end
lazy_map(f::Function, itr) = Map(f, itr)

function start(m :: Map) 
  s = start(m.itr)
  return s
end

function next(m :: Map, s) 
  n, ns = next(m.itr, s)
  return (m.flt(n), ns)
end

done(m :: Map, s) = done(m.itr, s)

function speech_frames(sf :: SegmentedFile)
  m, n = mask(sf)
  lazy_map(x -> x[2], filter(f -> m[f[1]], enumerate(HTKFeatures(sf.fn))))
end

function nonspeech_frames(sf :: SegmentedFile)
  m, n = mask(sf)
  lazy_map(x -> x[2], filter(f -> !m[f[1]], enumerate(HTKFeatures(sf.fn))))
end

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

function test(ana, dir, speech, nonspeech; threshold = 0.0)
  files     = analist(ana, dir = dir)
  
  N      = 0
  FAs    = 0
  misses = 0
  for sf in files
    speech_mask, speech_frames = mask(sf)
    for (i, f) in enumerate(HTKFeatures(sf.fn))
      score  = ll(speech, f) - ll(nonspeech, f)
      if (score < threshold) && speech_mask[i]
        misses += 1
      elseif (score >= threshold) && !speech_mask[i]
        FAs += 1
      end
      N += 1
    end
  end

  return FAs, misses, N
end

function optimize(ana, dir, speech, nonspeech; c_miss = 1.0, c_fa = 1.0)
  files  = analist(ana, dir = dir)
  scores = (Float32, Bool)[]

  N_speech = 0
  for sf in files
    speech_mask, speech_frames = mask(sf)
    for (i, f) in enumerate(HTKFeatures(sf.fn))
      p_speech    = ll(speech, f)
      p_nonspeech = ll(nonspeech, f)
      score       = p_speech - p_nonspeech
      push!(scores, (score, speech_mask[i]))
      if speech_mask[i]
        N_speech += 1
      end
    end
  end

  prev = 0.0
  
  N    = length(scores)
  N_ns = N - N_speech
  dc   = zeros(N)
  fa   = zeros(N)
  miss = zeros(N)
  hits = 0
  fas  = 0
  i    = 1

  for (score, speechp) in sort!(scores, by = x -> - x[1])
    if speechp
      hits += 1
    else
      fas += 1
    end

    far     = fas / float32(N_ns)
    missr   = (N_speech - hits) / float32(N_speech)
    fa[i]   = far
    miss[i] = missr
    dc[i]   = far * c_fa + missr * c_miss
    #@debug "$i -> $far $missr $(dc[i])"
    i      += 1
  end
  best = indmin(dc)
  
  return scores[best][1], fa[best], miss[best], dc[best], fa, miss
end

# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------

# train
if false
  speech, nonspeech = trn(ARGS[1], ARGS[3], splits = 7)
  outf = open("rats-sad.gmm", "w")
  serialize(outf, speech)
  serialize(outf, nonspeech)
  close(outf)
end

# score
f = open("rats-sad.gmm", "r")
speech    = deserialize(f)
nonspeech = deserialize(f)
close(f)

threshold, opt_fa, opt_miss, opt_dc, fa, miss = optimize(ARGS[2], ARGS[3], speech, nonspeech)
@info "optimum threshold = $threshold [fa rate = $opt_fa, miss rate = $opt_miss, decision cost = $opt_dc]"

FAs, misses, N = test(ARGS[2], ARGS[3], speech, nonspeech, threshold = threshold)
@info "FA rate   = $(FAs / N) (N = $FAs)"
@info "Miss rate = $(misses / N) (N = $misses)"
@info "N         = $N"

