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

void test_chat_formatting() {
    // this will hold the messages in the simulated 'chat'
    std::vector<wooly_chat_message> messages;
    messages.push_back({"system", "You are a creative writer."});
    messages.push_back({"user", "Write the start to the next sci-fi summer blockbuster movie in one paragraph and make sure to include a story hook."});


    // setup the model and context parameters for loading the model
    wooly_llama_model_params model_params = wooly_get_default_llama_model_params();
    model_params.n_gpu_layers = 100;
    wooly_llama_context_params context_params = wooly_get_default_llama_context_params();
    context_params.n_threads = 0;
    context_params.n_ctx = 1024 * 4;

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
    params.n_predict = 250;
    params.temp = 0.1f;
    params.top_k = 1;
    params.top_p = 1.0f;
    params.min_p = 0.03f;
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

    params.antiprompt_count = 0;
    params.antiprompts = NULL;

    // build up the prompt based on the formatting specified in the GGUF
    auto prompt_buffer_size = context_params.n_ctx * 10;
    char *prompt_buffer = (char*) calloc(prompt_buffer_size, sizeof(char));

    // we loop a few times to simulate a few chat rounds and hand the AI
    // some canned replies to whatever they generate.
    for (int chat_round=0; chat_round<3; ++chat_round) {
        printf("\033[33mUSER:\n%s\033[0m\n\n", messages.back().content);

        wooly_apply_chat_template(
            loaded_model.model, 
            NULL,
            true,
            messages.data(),
            messages.size(),
            prompt_buffer,
            prompt_buffer_size
        );
        params.prompt = prompt_buffer;
        //printf("DEBUG PROMPT:\n%s\n\n", prompt_buffer);

        // TODO: Further optimization of this loop can happen so that it can just
        // keep a running context without having to re-ingest the prompt each time
        // if enough features are added to the core API.

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
        

        // zero out our prediction token array
        int32_t *predicted_tokens = (int32_t*) calloc(params.n_predict, sizeof(int32_t));

        // run a prediction loop
        int32_t predicted = 0;
        wooly_sampler_t* sampler_ptr = results.gpt_sampler;
        while (predicted < params.n_predict) {
            predicted_tokens[predicted] = wooly_sample_next(loaded_model.ctx, sampler_ptr);
            
            // do all the antiprompt testing and eog testing
            int32_t eog = wooly_check_eog_and_antiprompt(params, loaded_model.ctx, loaded_model.model, sampler_ptr);
            if (eog > 0) {
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
        // NOTE: we never actually clean up the memory for `prediction_str` out of laziness...
        size_t prediction_buffer_size = (predicted+1) * 4 * 10;
        char *prediction_str = (char *) calloc(prediction_buffer_size, sizeof(char));
        int64_t pred_str_len = wooly_llama_detokenize(
            loaded_model.ctx, 
            false, 
            predicted_tokens, 
            predicted, 
            prediction_str, 
            prediction_buffer_size);
        TEST_ASSERT_GREATER_OR_EQUAL(0, pred_str_len);
        printf("\033[36mAI (tokens: %d):\n%s\033[0m\n\n", predicted, prediction_str);
        
        // put the generated message in the log
        messages.push_back({"assistant", prediction_str});

        // we have the 'user' reply in some generic ways with a fake, static response.
        const char* next_user_message;
        if (chat_round == 0) {
            next_user_message = "I don't even know what any of that means!! Can you give away more of the plot!";
        } else { 
            next_user_message = "Nope. I'm still lost! Can you try explaining it a different way?";
        }

        messages.push_back({"user", next_user_message});

        // clean up some of the memory
        wooly_free_sampler(sampler_ptr);
        free(predicted_tokens);
    }

    wooly_free_model(loaded_model.ctx, loaded_model.model);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_chat_formatting);
    return UNITY_END();
}