#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <string.h>
#include "unity.h"
#include "../bindings.h" 
#include "common.h"
#include "llama.h"

/*  =================================

    This test covers creating embeddings using specialized embedding models and is particuarlly
    tuned for `nomic-embed-text-v1.5`. It creates a series of embeddings and then calculates
    cosine similarities between a few example sentences and prints out the results.

    Additionally, this test covers the tokenization functions of woolycore, comparing the generated
    data against what is returned by the raw llama functions.

    =================================   */

void setUp(void) {}
void tearDown(void) {}

// yoinked from the embedding.cpp example in upstream llama.cpp
static void batch_add_seq(llama_batch & batch, const std::vector<int32_t> & tokens, llama_seq_id seq_id) {
    size_t n_tokens = tokens.size();
    for (size_t i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], (int32_t) i, { seq_id }, true);
    }
}

// yoinked from the embedding.cpp example in upstream llama.cpp
static void batch_decode_embeddings(llama_context * ctx, llama_batch & batch, float * output, int n_seq, int n_embd, int embd_norm) {
    const enum llama_pooling_type pooling_type = llama_pooling_type(ctx);
    const struct llama_model * model = llama_get_model(ctx);

    // clear previous kv_cache values (irrelevant for embeddings)
    llama_kv_cache_clear(ctx);

    // run model
    //fprintf(stderr, "%s: n_tokens = %d, n_seq = %d\n", __func__, batch.n_tokens, n_seq);
    if (llama_model_has_encoder(model) && !llama_model_has_decoder(model)) {
        // encoder-only model
        if (llama_encode(ctx, batch) < 0) {
            fprintf(stderr, "%s : failed to encode\n", __func__);
        }
    } else if (!llama_model_has_encoder(model) && llama_model_has_decoder(model)) {
        // decoder-only model
        if (llama_decode(ctx, batch) < 0) {
            fprintf(stderr, "%s : failed to decode\n", __func__);
        }
    }

    for (int i = 0; i < batch.n_tokens; i++) {
        if (!batch.logits[i]) {
            continue;
        }

        const float * embd = nullptr;
        int embd_pos = 0;

        if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
            // try to get token embeddings
            embd = llama_get_embeddings_ith(ctx, i);
            embd_pos = i;
            GGML_ASSERT(embd != NULL && "failed to get token embeddings");
        } else {
            // try to get sequence embeddings - supported only when pooling_type is not NONE
            embd = llama_get_embeddings_seq(ctx, batch.seq_id[i][0]);
            embd_pos = batch.seq_id[i][0];
            GGML_ASSERT(embd != NULL && "failed to get sequence embeddings");
        }

        float * out = output + embd_pos * n_embd;
        common_embd_normalize(embd, out, n_embd, embd_norm);
    }
}


// Function to get the test model path from environment variable
const char* get_test_model_path() {
    const char *model_filepath = getenv("WOOLY_TEST_EMB_MODEL_FILE");
    if (model_filepath != NULL) {
        return model_filepath;
    } else {
        printf("Set WOOLY_TEST_EMB_MODEL_FILE environment variable to the gguf embedding model to use for testing\n");
        exit(1);
    }
}

bool int32_arrays_equal(const int32_t* arr1, const int32_t* arr2, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (arr1[i] != arr2[i]) {
            return false;
        }
    }
    return true;
}

bool float_arrays_roughly_equal(const float* arr1, const float* arr2, size_t size, float epsilon) {
    for (size_t i = 0; i < size; ++i) {
        if (fabsf(arr1[i] - arr2[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

void test_embeddings() {
    // setup the model and context parameters for loading the model
    wooly_llama_model_params model_params = wooly_get_default_llama_model_params();
    model_params.n_gpu_layers = 100;
    wooly_llama_context_params context_params = wooly_get_default_llama_context_params();
    context_params.n_ctx = 2048; // make configurable
    context_params.n_batch = context_params.n_ctx; 
    context_params.n_ubatch = context_params.n_batch; // must be the same for non-causal models

    common_init();

    // crucial for the embeddings test: set this flag to true!
    context_params.embeddings = true;

    // for our test we'll use llama_pooling_type::LLAMA_POOLING_TYPE_MEAN pooling
    context_params.pooling_type = 1;

    // use euclidian normalization for the embedding vectors
    const int32_t embd_normalize = 2;


    // get the model filepath from the environment variable and load it up
    const char *model_filepath = get_test_model_path();
    wooly_load_model_result loaded_model = wooly_load_model(model_filepath, model_params, context_params, true);

    // check to make sure we got pointers for the model and the context
    TEST_ASSERT_NOT_NULL(loaded_model.model);
    TEST_ASSERT_NOT_NULL(loaded_model.ctx);


    // setup some test sentences to test for similarity against the first prompt
    std::vector<std::string> prompts = {
        "That is a happy person.", 
        "That's a very happy person.",
        "She is not happy about the news.",
        "Is that a happy dog?",
        "Behold an individual brimming with boundless joy and contentment.",
        "The weather is beautiful today.",
        "The sun is shining brightly.",
        "I've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion.", 
    };

    // allocate the buffer that we'll use for our C-compatible API.
    const auto token_buffer_size = context_params.n_ctx;
    printf("Allocating token buffer of size %d\n", token_buffer_size);
    int32_t* token_buffer = static_cast<int32_t*>(malloc(sizeof(int32_t) * token_buffer_size));

    // tokenize the prompts and check to make sure the raw llama tokenize method
    // returns the same data as our own.
    std::vector<std::vector<int32_t>> tokenized_prompts;
    for (const auto & prompt : prompts) {
        // we do the tokenization with the original llama call
        auto llama_tokens = common_tokenize(static_cast<const llama_context *>(loaded_model.ctx), prompt.c_str(), true, true);

        // and then use our wrapped library
        size_t num_of_tokens = wooly_llama_tokenize(
            loaded_model.ctx, 
            prompt.c_str(), 
            true, 
            false,
            token_buffer,
            token_buffer_size);

        // compare the results to make sure that we're generating the same tokens
        TEST_ASSERT_LESS_OR_EQUAL(token_buffer_size, num_of_tokens);
        TEST_ASSERT_EQUAL_INT32(llama_tokens.size(), num_of_tokens);
        TEST_ASSERT_TRUE(int32_arrays_equal(llama_tokens.data(), token_buffer, num_of_tokens));
        printf("Prompt tokenized to %zu tokens: %s\n", num_of_tokens, prompt.c_str());

        // copy the data out of our buffer and store it into a vector of token vectors.
        std::vector<int32_t> token_vector;
        token_vector.resize(num_of_tokens);
        std::copy(token_buffer, token_buffer + num_of_tokens, token_vector.begin());
        tokenized_prompts.push_back(token_vector);

        // do a detokenize pass on the prompt tokens to test that API call
        size_t detoken_buffer_size = context_params.n_ctx;
        char* detoken_buffer = static_cast<char*>(malloc(detoken_buffer_size));
        int64_t detoken_count = wooly_llama_detokenize(
            loaded_model.ctx, 
            false, 
            token_buffer, 
            num_of_tokens, 
            detoken_buffer,
            detoken_buffer_size);
        printf("\tDetokenized: %s\n", detoken_buffer);

        // testing the equality of the detokenized text isn't uniformly supported. for example,
        // the nomic embedding model this test was designed originally to use does not round
        // trip the tokens to the exact same text. But if you run this test using Llama-3.1,
        // you'll see that the text round trips perfectly. for that reason, the test is
        // currently disabled.
        //
        // bool detoken_is_same = std::equal(prompt.begin(), prompt.end(), detoken_buffer, [](char c1, char c2) {
        //     return std::toupper(c1) == std::toupper(c2);
        // });
        // if (detoken_count != prompt.length()) {
        //     detoken_is_same = false;
        // }
        // TEST_ASSERT_TRUE(detoken_is_same);

        free(detoken_buffer);
    }

    // setup the batch process for the raw llama API
    const int n_prompts = (int) prompts.size();

    // count number of embeddings; if no pooling, then each token has its own
    // embedding vector, otherwise all of the embeddings for a given prompt are processed and
    // reduced down to one embedding vector.
    // n_embd_count ends up being the total number of embedding vectors needed for the output.
    int n_embd_count = 0;
    if (context_params.pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int k = 0; k < n_prompts; k++) {
            n_embd_count += (int) tokenized_prompts[k].size();
        }
    } else {
        n_embd_count = n_prompts;
    }

    // allocate output
    const int n_embd = wooly_llama_n_embd(loaded_model.model);
    printf("Embedding model size: %d\n", n_embd);
    std::vector<float> embeddings(n_embd_count * n_embd, 0);
    float * emb = embeddings.data();

    // break into batches
    struct llama_batch batch = llama_batch_init(context_params.n_batch, 0, 1);
    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < n_prompts; k++) {
        // clamp to n_batch tokens
        auto & inp = tokenized_prompts[k];
        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > context_params.n_batch) {
            float * out = emb + e * n_embd;
            batch_decode_embeddings(static_cast<llama_context *>(loaded_model.ctx), batch, out, s, n_embd, embd_normalize);
            e += context_params.pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch - after this our embeddings vector should be fully populated
    float * out = emb + e * n_embd;
    batch_decode_embeddings(static_cast<llama_context *>(loaded_model.ctx), batch, out, s, n_embd, embd_normalize);

    // **** Switch over to testing the Woolycore API ****

    // calculate the size of the output buffer we'll need
    auto total_embd_floats = n_embd * n_prompts; // we're always doing pooling, so don't worry about the non-pooling case
    float* wooly_embd_buffer = static_cast<float*>(malloc(sizeof(float) * total_embd_floats));

    // we're gonna pass in the pointers to the tokenized arrays, so create that...
    // we can't use STL collections with a C-compatible API so we have to make a copy into temporary buffers
    std::vector<int32_t*> token_array_ptrs;
    std::vector<int64_t> token_array_sizes;
    for (size_t k = 0 ; k < tokenized_prompts.size(); ++k) {
        auto token_array = tokenized_prompts[k];
        auto token_array_size = static_cast<int64_t>(token_array.size());
        
        int32_t* token_array_buffer = static_cast<int32_t*>(malloc(sizeof(int32_t) * token_array_size));
        std::copy(token_array.begin(), token_array.end(), token_array_buffer);

        token_array_ptrs.push_back(token_array_buffer);
        token_array_sizes.push_back(token_array_size);
    }

    // generate our embeddings with our API
    auto embd_ret = wooly_llama_make_embeddings(
        loaded_model.model,
        loaded_model.ctx,
        context_params.n_batch,
        context_params.pooling_type,
        embd_normalize,
        token_array_ptrs.size(),
        token_array_ptrs.data(),
        token_array_sizes.data(),
        wooly_embd_buffer,
        total_embd_floats);

    TEST_ASSERT_EQUAL(0, embd_ret);
    TEST_ASSERT_TRUE(float_arrays_roughly_equal(emb, wooly_embd_buffer, total_embd_floats, 0.0001f));

    // free up our raw buffers we sent to the API
    for (auto i=0; i<token_array_ptrs.size(); ++i) {
        free(token_array_ptrs[i]);
    }


    // print the first part of the embeddings or for a single prompt, the full embedding
    for (int j = 0; j < n_prompts; j++) {
        printf("embedding %d: ", j);
        for (int i = 0; i < (n_prompts > 1 ? std::min(16, n_embd) : n_embd); i++) {
            if (embd_normalize == 0) {
                printf("%6.0f ", emb[j * n_embd + i]);
            } else {
                printf("%9.6f ", emb[j * n_embd + i]);
            }
        }
        printf("\n");
    }

    /// print cosine similarity matrix
    if (n_prompts > 1) {
        printf("\n");
        printf("cosine similarity matrix:\n\n");
        for (int i = 0; i < n_prompts; i++) {
            printf("%6.6s ", prompts[i].c_str());
        }
        printf("\n");
        for (int i = 0; i < n_prompts; i++) {
            for (int j = 0; j < n_prompts; j++) {
                float sim = common_embd_similarity_cos(emb + i * n_embd, emb + j * n_embd, n_embd);
                printf("%6.2f ", sim);
            }
            printf("%s", prompts[i].c_str());
            printf("\n");
        }
    }


    llama_batch_free(batch);
    wooly_free_model(loaded_model.ctx, loaded_model.model);
}

int main() {
    UNITY_BEGIN();
    RUN_TEST(test_embeddings);
    return UNITY_END();
}