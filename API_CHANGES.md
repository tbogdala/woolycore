# API Changes

## v0.3.0

* The `wooly_predict_result` struct had `t_load_us` removed since that was only relevant during model loading.
* If `wooly_predict()` uses the `prompt_cache_ptr` param, `n_p_eval` will be 0.
* The `wooly_gpt_params` struct had `tfs_z` removed since it was removed upstream.


## v0.2.0

* The `wooly_predict_result` struct had `t_sample_ms` and `n_sample` removed from it due to upstream changes.
* The `wooly_llama_context_params` struct had `seed` removed as it's non-functional anyway.
* The upstream changes for renaming things as 'common' instead of 'gpt' should be invisible in practice,
  but there are parameter name changes for 'sampler' parameters.
* Changed `wooly_llama_make_embeddings` to return `int64_t` for consistency instead of `long`.