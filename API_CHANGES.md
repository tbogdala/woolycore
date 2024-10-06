# API Changes

* The `wooly_predict_result` struct had `t_sample_ms` and `n_sample` removed from it due to upstream changes.
* The `wooly_llama_context_params` struct had `seed` removed as it's non-functional anyway.