# API Changes

## v0.2.0

* The `wooly_predict_result` struct had `t_sample_ms` and `n_sample` removed from it due to upstream changes.
* The `wooly_llama_context_params` struct had `seed` removed as it's non-functional anyway.
* The upstream changes for renaming things as 'common' instead of 'gpt' should be invisible in practice,
  but there are parameter name changes for 'sampler' parameters.