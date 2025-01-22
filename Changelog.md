# Change log

## v0.4.0

* BREAKING: All `void*` parameters got type aliases to empty structs to provide some type safety
  for the external API. This change has been brewing for some time and is the only major break
  for the near future.

* feature: New `wooly_has_chat_template()` and `wooly_apply_chat_template` which takes a list of
  the new `wooly_chat_message` struct and uses the embedded template inside the GGUF model to 
  build a prompt with the list of messages supplied.

* feature: New unit test called `test_chat_formatting` to test the new embedded chat template code.

* breaking: `penalize_nl` was removed from `wooly_gpt_params` as it was removed upstream in llama.cpp.


## v0.3.0

* feature: added `wooly_process_additional_prompt()` to allow for adding more prompt text to
  a context that has already called `wooly_process_prompt()`.

* bug: reverted `wooly_predict()` behavior so that when the prompt cache is used, the number of
  prompt tokens evaluated will be 0 and not 1 (originally changed to 1 for upstream llama.cpp compatibility).
  
* maintenance: rewrote `wooly_predict()` in terms of the stepwise functions `wooly_process_prompt()`
  `wooly_sample_next()` and `wooly_process_next_token()`, etc... should be more maintainable.