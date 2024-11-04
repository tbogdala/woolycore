# Change log

## v0.3.0

* bug: reverted `wooly_predict()` behavior so that when the prompt cache is used, the number of
  prompt tokens evaluated will be 0 and not 1 (originally changed to 1 for upstream llama.cpp compatibility).
  
* maintenance: rewrote `wooly_predict()` in terms of the stepwise functions `wooly_process_prompt()`
  `wooly_sample_next()` and `wooly_process_next_token()`, etc... should be more maintainable.