# Change log

## v0.3.0

* feature: added `wooly_process_additional_prompt()` to allow for adding more prompt text to
  a context that has already called `wooly_process_prompt()`.

* bug: reverted `wooly_predict()` behavior so that when the prompt cache is used, the number of
  prompt tokens evaluated will be 0 and not 1 (originally changed to 1 for upstream llama.cpp compatibility).
  
* maintenance: rewrote `wooly_predict()` in terms of the stepwise functions `wooly_process_prompt()`
  `wooly_sample_next()` and `wooly_process_next_token()`, etc... should be more maintainable.