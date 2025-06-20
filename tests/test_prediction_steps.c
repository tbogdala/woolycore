#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "unity.h"
#include "../bindings.h" 

/*  =================================

    This unit test is designed to test the iterative prediction loop with lower level functions
    than just `wooly_predict()`. Additionally it tests caching the prediction state after
    ingesting a prompt AND after doing a prediction.

    By running this, you should see three different blocks of predicted text. The first
    block uses very static sampler settings. The second block reuses the cached prompt state
    but with different settings, so you should see a different answer. The third block
    of text tests continuing the first prediction, so it should be a continuation of the
    first block and flow well together.

    =================================   */

void setUp(void) {}
void tearDown(void) {}

// Simple callback to print the string out and flush the output stream.
bool predict_callback(const char *token_str) {
    printf("%s", token_str);
    fflush(stdout);
    return true;
}

// Function to get the test model path from environment variable
const char* get_test_model_path() {
    const char *model_filepath = getenv("WOOLY_TEST_MODEL_FILE");
    if (model_filepath != NULL) {
        return model_filepath;
    } else {
        printf("Set WOOLY_TEST_MODEL_FILE environment variable to the gguf file to use for testing\n");
        exit(1);
    }
}

void test_predictions() {
    // setup the model and context parameters for loading the model
    wooly_llama_model_params model_params = wooly_get_default_llama_model_params();
    model_params.n_gpu_layers = 100;
    wooly_llama_context_params context_params = wooly_get_default_llama_context_params();
    context_params.n_threads = -1;
    context_params.n_ctx = 1024 * 2;

    // get the model filepath from the environment variable and load it up
    const char *model_filepath = get_test_model_path();
    wooly_load_model_result loaded_model = wooly_load_model(model_filepath, model_params, context_params, true);

    // check to make sure we got pointers for the model and the context
    TEST_ASSERT_NOT_NULL(loaded_model.model);
    TEST_ASSERT_NOT_NULL(loaded_model.ctx);

    // setup the text prediction parameters
    struct wooly_gpt_params params = wooly_new_gpt_params();
    params.seed = 42;
    params.n_threads = -1;
    params.n_predict = 100;
    params.temp = 0.1f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.min_p = 0.1f;
    params.penalty_repeat = 1.1f;
    params.penalty_last_n = 512;
    params.ignore_eos = false;
    params.flash_attn = true;
    params.n_batch = 128;
    params.prompt_cache_all = false;

    params.dry_multiplier = 0.8;
    params.dry_base = 1.75;
    params.dry_allowed_length = 2;
    params.dry_penalty_last_n = -1;
    const char *dry_breakers[] = {"\n", ":", "\"", "*"};
    params.dry_sequence_breakers_count = 4;
    params.dry_sequence_breakers = dry_breakers;

    const char *antiprompts_array[] = {"<|end|>"};
    params.antiprompt_count = 1;
    params.antiprompts = antiprompts_array;
    params.prompt = "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n";


    // get the prompt ingested into the context and pull the sampler
    // used in the process so that repeat penalties and such are
    // accounted for.
    wooly_process_prompt_results results = wooly_process_prompt(
        params, 
        loaded_model.ctx, 
        loaded_model.model);
    int32_t prompt_token_count = results.result;
    TEST_ASSERT_GREATER_OR_EQUAL(1, prompt_token_count);
    TEST_ASSERT_NOT_NULL(results.gpt_sampler);
    printf("\nPrompt token count processed: %d\n\n", prompt_token_count);


    // freeze the state of the context after processing the prompt so that
    // we may dethaw it repeatedly to remove prompt processing penalty for
    // multiple generations.
    wooly_prompt_cache_t* prompt_cache = wooly_freeze_prediction_state(
        params, 
        loaded_model.ctx, 
        loaded_model.model,
        NULL,
        0);
    TEST_ASSERT_NOT_NULL(prompt_cache);


    // zero out our prediction token array
    int32_t *predicted_tokens = calloc(params.n_predict, sizeof(int32_t));

    // run a prediction loop
    int32_t predicted = 0;
    wooly_sampler_t* sampler_ptr = results.gpt_sampler;
    while (predicted < params.n_predict) {
        predicted_tokens[predicted] = wooly_sample_next(loaded_model.ctx, sampler_ptr);
        
        // do all the antiprompt testing and eog testing
        int32_t eog = wooly_check_eog_and_antiprompt(params, loaded_model.ctx, loaded_model.model, sampler_ptr);
        if (eog > 0) {
            printf("End of generation token or antiprompt token encountered; stopping generation...\n");
            break;
        }

        // calculate the next logits (expensive compute)
        int32_t success = wooly_process_next_token(
            loaded_model.ctx, 
            predicted_tokens[predicted]);
        TEST_ASSERT_EQUAL(0, success);

        predicted++;
    }


    // convert our predicted tokens to a string and print the result
    size_t prediction_buffer_size = (predicted+1) * 4 * 10;
    char *first_prediction_str = calloc(prediction_buffer_size, sizeof(char));
    int64_t pred_str_len = wooly_llama_detokenize(
        loaded_model.ctx, 
        false, 
        predicted_tokens, 
        predicted, 
        first_prediction_str, 
        prediction_buffer_size);
    TEST_ASSERT_GREATER_OR_EQUAL(0, pred_str_len);
    printf("\nPrediction (tokens: %d):\n\n%s\n\n", predicted, first_prediction_str);


    // freeze the state again after the text prediction
    wooly_prompt_cache_t* first_prediction_cache = wooly_freeze_prediction_state(
        params, 
        loaded_model.ctx, 
        loaded_model.model,
        predicted_tokens,
        predicted);
    TEST_ASSERT_NOT_NULL(first_prediction_cache);
    int32_t first_prediction_count = predicted;

    wooly_free_sampler(sampler_ptr);

    /* ===== Prompt cache test ===== */

    puts("\n~~~ ---- ~~~~\n\n");

    memset(predicted_tokens, 0, sizeof(int32_t) * params.n_predict);

    // restore our prediction state to what it was after we had ingested the prompt
    wooly_process_prompt_results defrost_results = wooly_defrost_prediction_state(
        params,
        loaded_model.ctx, 
        loaded_model.model, 
        prompt_cache);

    // make sure we get the same number of tokens back after defrosting
    TEST_ASSERT_EQUAL(prompt_token_count, defrost_results.result);

    // use the new sampler to pickup on the adjusted settings
    sampler_ptr = defrost_results.gpt_sampler;

    // inject a little more prompt in just to make it different and test
    // the ability to add more prompt to ingest. This should produce a 
    // distinctly different result than the first prediction.
    const char* new_prompt_text = "Do you have a suggestion for genre?<|end|>\n<|user|>\nMake it like a Pixar movie script, but with those two authors!<|end|>\n<|assistant|>\n";
    int32_t additional_tokens = wooly_process_additional_prompt(
        loaded_model.ctx, 
        loaded_model.model,
        sampler_ptr,
        new_prompt_text
    );
    TEST_ASSERT_GREATER_THAN(0, additional_tokens);

    // run another prediction loop
    predicted = 0;
    while (predicted < params.n_predict) {
        predicted_tokens[predicted] = wooly_sample_next(loaded_model.ctx, sampler_ptr);
        
        // do all the antiprompt testing and eog testing
        int32_t eog = wooly_check_eog_and_antiprompt(params, loaded_model.ctx, loaded_model.model, sampler_ptr);
        if (eog > 0) {
            printf("End of generation token or antiprompt token encountered; stopping generation...\n");
            break;
        }

        // calculate the next logits (expensive compute)
        int32_t success = wooly_process_next_token(
            loaded_model.ctx, 
            predicted_tokens[predicted]);
        TEST_ASSERT_EQUAL(0, success);

        predicted++;
    }


    // convert our predicted tokens to a string and print the result
    prediction_buffer_size = (predicted+1) * 4 * 10;
    char* prediction_str = calloc(prediction_buffer_size, sizeof(char));
    pred_str_len = wooly_llama_detokenize(
        loaded_model.ctx, 
        false, 
        predicted_tokens, 
        predicted, 
        prediction_str, 
        prediction_buffer_size);
    TEST_ASSERT_GREATER_OR_EQUAL(0, pred_str_len);
    printf("\nPrediction (tokens: %d):\n\n%s\n\n", predicted, prediction_str);
    free(prediction_str);

    wooly_free_prompt_cache(prompt_cache);
    wooly_free_sampler(sampler_ptr);


    /* ===== Prompt cache test - continue mode ===== */

    puts("\n~~~ ---- ~~~~\n\n");

    memset(predicted_tokens, 0, sizeof(int32_t) * params.n_predict);

    // restore our prediction state to what it was after we had run our first prediction.
    // this should mean that the additional prompt string we added above will no longer
    // be there and the prediction can continue as it would have initially.
    defrost_results = wooly_defrost_prediction_state(
        params,
        loaded_model.ctx, 
        loaded_model.model, 
        first_prediction_cache);

    // make sure we get the same number of tokens back after defrosting
    TEST_ASSERT_EQUAL(prompt_token_count+first_prediction_count, defrost_results.result);

    // use the new sampler to pickup on the adjusted settings
    sampler_ptr = defrost_results.gpt_sampler;

    // run another prediction loop
    predicted = 0;
    while (predicted < params.n_predict) {
        predicted_tokens[predicted] = wooly_sample_next(loaded_model.ctx, sampler_ptr);
        
        // do all the antiprompt testing and eog testing
        int32_t eog = wooly_check_eog_and_antiprompt(params, loaded_model.ctx, loaded_model.model, sampler_ptr);
        if (eog > 0) {
            printf("End of generation token or antiprompt token encountered; stopping generation...\n");
            break;
        }

        // calculate the next logits (expensive compute)
        int32_t success = wooly_process_next_token(
            loaded_model.ctx, 
            predicted_tokens[predicted]);
        TEST_ASSERT_EQUAL(0, success);

        predicted++;
    }

    // convert our predicted tokens to a string and print the result.
    // Note: It should make sense as a continuation of the first result,
    // but this takes human eyes to notice (right now, at least ;p)!
    prediction_buffer_size = (predicted+1) * 4 * 10;
    prediction_str = calloc(prediction_buffer_size, sizeof(char));
    pred_str_len = wooly_llama_detokenize(
        loaded_model.ctx, 
        false, 
        predicted_tokens, 
        predicted, 
        prediction_str, 
        prediction_buffer_size);
    TEST_ASSERT_GREATER_OR_EQUAL(0, pred_str_len);
    printf("\nFinal Prediction (new tokens: %d):\n\n%s%s\n\n", predicted, first_prediction_str, prediction_str);
    free(first_prediction_str);
    free(prediction_str);
    free(predicted_tokens);

    wooly_free_prompt_cache(first_prediction_cache);
    wooly_free_sampler(sampler_ptr);
    wooly_free_model(loaded_model.ctx, loaded_model.model);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_predictions);
    return UNITY_END();
}