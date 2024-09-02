#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "unity.h"
#include "../bindings.h" 
#include "llama.h"

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
    context_params.seed = 42;
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
    params.n_threads = 4;
    params.n_predict = 100;
    params.temp = 0.1;
    params.top_k = 1;
    params.top_p = 1.0;
    params.min_p = 0.1;
    params.penalty_repeat = 1.1;
    params.penalty_last_n = 512;
    params.ignore_eos = false;
    params.flash_attn = true;
    params.n_batch = 128;
    params.prompt_cache_all = true;

    const char *antiprompts_array[] = {"<|end|>"};
    params.antiprompt_count = 1;
    params.antiprompts = antiprompts_array;
    params.prompt = "<|user|>\nWrite the start to the next movie collaboration between Quentin Tarantino and Robert Rodriguez.<|end|>\n<|assistant|>\n";

    // allocate an output buffer for the text prediction
    char *prediction = malloc((params.n_predict+1) * 4);
    wooly_predict_result results = wooly_predict(params, loaded_model.ctx, loaded_model.model, false, prediction, NULL, NULL);

    // ensure the result code was 0 to indicate no errors happened and print out prediction and timing info
    TEST_ASSERT_EQUAL(0, results.result);
    TEST_ASSERT_GREATER_OR_EQUAL(1, results.n_eval);
    printf("\n%s\n\nTiming Data: %d tokens total in %.2f ms (%.2f T/s) ; %d prompt tokens in %.2f ms (%.2f T/s)\n\n",
           prediction,
           results.n_eval,
           (results.t_end_ms - results.t_start_ms),
           1e3 / (results.t_end_ms - results.t_start_ms) * results.n_eval,
           results.n_p_eval,
           results.t_p_eval_ms,
           1e3 / results.t_p_eval_ms * results.n_p_eval);

    /* ===== Prompt cache test ===== */

    // change only the seed and use the prompt_cache from the results above. this should cause the prediction
    // to skip the prompt processing and head straight for text prediction. additionally we use the callback
    // here so text will be output realtime to stdout.
    params.seed = 1337;
    results = wooly_predict(params, loaded_model.ctx, loaded_model.model, false, prediction, results.prompt_cache, predict_callback);

    // ensure no errors happened and ensure the number of prompt tokens processed is 0 since we used the prompt_cache
    TEST_ASSERT_EQUAL(0, results.result);
    TEST_ASSERT_EQUAL(0, results.n_p_eval);
    TEST_ASSERT_GREATER_OR_EQUAL(1, results.n_eval);
    printf("\n\nTiming Data: %d tokens total in %.2f ms (%.2f T/s) ; %d prompt tokens in %.2f ms (%.2f T/s)\n\n",
           results.n_eval,
           (results.t_end_ms - results.t_start_ms),
           1e3 / (results.t_end_ms - results.t_start_ms) * results.n_eval,
           results.n_p_eval,
           results.t_p_eval_ms,
           1e3 / results.t_p_eval_ms * results.n_p_eval);

    free(prediction);

    /* ===== Grammar test ===== */

    // this test uses the upstream json grammar file to force the text prediction to follow JSON grammar.
    // here, we load it from the text file and allocate a string to hold it.
    FILE *fp = fopen("llama.cpp/grammars/json.gbnf", "r");
    if (fp == NULL) {
        printf("Couldn't read the GBNF grammar file from upstream llama.cpp\n");
        exit(1);
    }
    fseek(fp, 0, SEEK_END);
    long gbnf_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *gbnf_string = malloc(gbnf_size + 1);
    fread(gbnf_string, 1, gbnf_size, fp);
    fclose(fp);
    gbnf_string[gbnf_size] = '\0';

    // set the number of tokens to predict to -1 so that it goes until the job's done; that does impact
    // how big we want to allocate our prediction buffer, so that will be big enough to hold the whole context
    // as a worst case.
    prediction = malloc(context_params.n_ctx * 4);
    params.n_predict = -1;
    params.prompt = "<|user|>\nReturn a JSON object that describes an object in a fictional Dark Souls game. The returned JSON object should have 'Title' and 'Description' fields that define the item in the game. Make sure to write the item lore in the style of Fromsoft and thier Dark Souls series of games: there should be over-the-top naming of fantastically gross monsters and tragic historical events from the world, all with a very nihilistic feel.<|end|>\n<|assistant|>\n";
    
    // set the grammar field of the structure to the grammar file we loaded to trigger llama.cpp's grammar support
    params.grammar = gbnf_string;
    results = wooly_predict(params, loaded_model.ctx, loaded_model.model, false, prediction, NULL, predict_callback);
    
    // ensure no errors happened and make sure the prompt was processed
    TEST_ASSERT_EQUAL(0, results.result);
    TEST_ASSERT_NOT_EQUAL(0, results.n_p_eval);
    TEST_ASSERT_GREATER_OR_EQUAL(1, results.n_eval);
    printf("\n\nTiming Data: %d tokens total in %.2f ms (%.2f T/s) ; %d prompt tokens in %.2f ms (%.2f T/s)\n\n",
           results.n_eval,
           (results.t_end_ms - results.t_start_ms),
           1e3 / (results.t_end_ms - results.t_start_ms) * results.n_eval,
           results.n_p_eval,
           results.t_p_eval_ms,
           1e3 / results.t_p_eval_ms * results.n_p_eval);

    free(gbnf_string);
    free(prediction);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_predictions);
    return UNITY_END();
}