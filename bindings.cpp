#include "llama.h"
#include "common/common.h"
#include "log.h"
#include "sampling.h"
#include "bindings.h"

#include <regex>
#include <fstream>
#include <string>
#include <cstring>
#include <vector>

// internal functions headers
void 
fill_params_from_simple(
    wooly_gpt_params *simple, 
    struct common_params *output);

llama_model_params
conv_wooly_to_llama_model_params(
    wooly_llama_model_params wooly_params);

llama_context_params
conv_wooly_to_llama_context_params(
    wooly_llama_context_params wooly_params);



static void 
llama_log_callback_silent(
    ggml_log_level level, 
    const char * text, 
    void * user_data) 
{
    // do nothing. :D
}

typedef struct llama_predict_prompt_cache 
{
    std::string last_prompt;
    std::vector<llama_token> processed_prompt_tokens;
    uint8_t * last_processed_prompt_state;
    size_t last_processed_prompt_state_size;
} llama_predict_prompt_cache;

static bool 
file_exists(
    const std::string & path) 
{
    std::ifstream f(path.c_str());
    return f.good();
}

static bool 
file_is_empty(
    const std::string & path) 
{
    std::ifstream f;
    f.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    f.open(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
    return f.tellg() == 0;
}

static std::vector<std::string> 
create_vector(
    const char **strings, 
    int count)
{
    std::vector<std::string> *vec = new std::vector<std::string>;
    for (int i = 0; i < count; i++)
    {
        vec->push_back(std::string(strings[i]));
    }
    return *vec;
}

static void 
delete_vector(
    std::vector<std::string> *vec)
{
    delete vec;
}

wooly_load_model_result 
wooly_load_model(
    const char *fname, 
    wooly_llama_model_params wooly_model_params, 
    wooly_llama_context_params wooly_context_params,
    bool silent_llama)
{
    if (silent_llama) {
        llama_log_set(llama_log_callback_silent, NULL);
        common_log_pause(common_log_main());
    } else {
#ifdef WOOLY_DEBUG
        common_log_set_verbosity_thold(1); 
#else
        common_log_set_verbosity_thold(0); 
#endif
    }

    // load dynamic backends
    ggml_backend_load_all();
    
    llama_model *model = nullptr;
    llama_context * lctx = nullptr;
    wooly_load_model_result res;
    res.ctx = nullptr;
    res.model = nullptr;
    // TODO: implement lora adapters (e.g. llama_init_from_gpt_params())

    llama_model_params model_params = conv_wooly_to_llama_model_params(wooly_model_params);
    llama_context_params context_params = conv_wooly_to_llama_context_params(wooly_context_params);
    try
    {
        model = llama_model_load_from_file(fname, model_params);
        if (model == NULL) {
            return res;
        }
	    lctx = llama_init_from_model(model, context_params);
        if (lctx == NULL) {
            llama_model_free(model);
            return res;
        }
    }
    catch (std::runtime_error &e)
    {
        LOG_ERR("failed %s", e.what());
        llama_free(lctx);
        llama_model_free(model);
        return res;
    }

    {
        LOG_WRN("warming up the model with an empty run\n");

        const llama_vocab *vocab = llama_model_get_vocab(model);
        std::vector<llama_token> tmp;
        llama_token bos = llama_vocab_bos(vocab);
        llama_token eos = llama_vocab_eos(vocab);
        // some models (e.g. T5) don't have a BOS token
        if (bos != LLAMA_TOKEN_NULL) {
            tmp.push_back(bos);
        }
        if (eos != LLAMA_TOKEN_NULL) {
            tmp.push_back(eos);
        }
        if (tmp.empty()) {
            tmp.push_back(0);
        }

        if (llama_model_has_encoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), (int32_t) tmp.size()));
            llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
            if (decoder_start_token_id == -1) {
                decoder_start_token_id = bos;
            }
            tmp.clear();
            tmp.push_back(decoder_start_token_id);
        }
        if (llama_model_has_decoder(model)) {
            llama_decode(lctx, llama_batch_get_one(tmp.data(), std::min((int32_t) tmp.size(), (int32_t) context_params.n_batch)));
        }
        llama_kv_cache_clear(lctx);
        llama_synchronize(lctx);
        llama_perf_context_reset(lctx);
    }

    res.ctx = (wooly_llama_context_t*) lctx;
    res.model = (wooly_llama_model_t*) model;
    res.context_length = llama_n_ctx(lctx);
    return res;
}

void 
wooly_free_model(
    wooly_llama_context_t* llama_context_ptr, 
    wooly_llama_model_t* llama_model_ptr)
{
    llama_context *ctx = (llama_context *) llama_context_ptr; 
    llama_model *model = (llama_model *) llama_model_ptr;
    if (model != NULL) {
        llama_model_free(model);
    }
    if (ctx != NULL) {
        llama_free(ctx);
    }
}

void 
wooly_free_sampler(
    wooly_sampler_t* sampler_ptr)
{
    if (sampler_ptr != NULL) {
        common_sampler *smpl = (common_sampler *)(sampler_ptr);
        common_sampler_free(smpl);
    }
}


int32_t
wooly_process_additional_prompt(
    wooly_llama_context_t*  llama_context_ptr, 
    wooly_llama_model_t*    llama_model_ptr,
    wooly_sampler_t*        sampler_ptr,
    const char*             additional_prompt)
{
    // sanity check additional prompt input string
    if (additional_prompt == NULL || strlen(additional_prompt) <= 0) {
        return -1;
    }

    // certain things that are done in `wooly_process_prompt()` won't be done here,
    // which include some sanity warnings, KV cache reset, threadpool initialization
    // or change, dealing with BOS token additon, checking for context overflow,
    // sampler initialization ...

    llama_context *ctx = (llama_context *)llama_context_ptr; 
    llama_model *model = (llama_model *)llama_model_ptr;
    common_sampler *smpl = (common_sampler *)sampler_ptr;

    // tokenize the additional prompt text
    std::vector<llama_token> prompt_tokens;
    LOG_DBG("tokenize the additional prompt\n");
    prompt_tokens = ::common_tokenize(ctx, additional_prompt, false, true);
    LOG_DBG("additional prompt: \"%s\"\n", additional_prompt);
    LOG_DBG("tokens: %s\n", string_from(ctx, prompt_tokens).c_str());

    // if we came up with no extra tokens just return
    if (prompt_tokens.empty()) {
        LOG_DBG("additional prompt yielded no tokens; returning ..\n");
        return 0;
    }

    // Ingest the prompt
    int32_t n_consumed = 0;
    while ((int)prompt_tokens.size() > n_consumed) {
        // push the prompt in the sampling context in order to apply repetition penalties later
        // for the prompt, we don't apply grammar rules
        common_sampler_accept(smpl, prompt_tokens[n_consumed], /* accept_grammar= */ false);
        ++n_consumed;
    }

    uint32_t n_batch = llama_n_batch(ctx);
    n_consumed = 0;
    for (int i = 0; i < (int)prompt_tokens.size(); i += n_batch) {
        int n_eval = (int)prompt_tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx, llama_batch_get_one(&prompt_tokens[i], n_eval))) {
            LOG_ERR("%s : failed to eval batch of %d at offset %d\n", __func__, n_eval, i);
            return -2;
        }
        n_consumed += n_eval;
        LOG_DBG("%s: prompt tokens consumed so far = %d / %zu\n", __func__, n_consumed, prompt_tokens.size());
    }

    return n_consumed;
}

wooly_process_prompt_results 
wooly_process_prompt(
    wooly_gpt_params        simple_params, 
    wooly_llama_context_t*  llama_context_ptr, 
    wooly_llama_model_t*    llama_model_ptr) 
{
    wooly_process_prompt_results return_value;
    llama_context *ctx = (llama_context *)llama_context_ptr; 
    llama_model *model = (llama_model *)llama_model_ptr;

    common_params params;
    fill_params_from_simple(&simple_params, &params);


    // Do a basic set of warnings based on incoming parameters
    const int n_ctx_train = llama_model_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }
    if (params.n_ctx != 0 && params.n_ctx < 8) {
        LOG_WRN("%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }
    if (params.rope_freq_base != 0.0) {
        LOG_WRN("%s: warning: changing RoPE frequency base to %g.\n", __func__, params.rope_freq_base);
    }
    if (params.rope_freq_scale != 0.0) {
        LOG_WRN("%s: warning: scaling RoPE frequency by %g.\n", __func__, params.rope_freq_scale);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }


    // Reset KV Cache and timing data
    llama_kv_cache_clear(ctx);
    llama_perf_context_reset(ctx);


    // Setup the threadpool for this processing task
    LOG_INF("%s: llama threadpool init = n_threads = %d\n", __func__, (int) params.cpuparams.n_threads);
    struct ggml_threadpool_params tpp_batch =
            ggml_threadpool_params_from_cpu_params(params.cpuparams_batch);
    struct ggml_threadpool_params tpp =
            ggml_threadpool_params_from_cpu_params(params.cpuparams);

    set_process_priority(params.cpuparams.priority);

    struct ggml_threadpool * threadpool_batch = NULL;
    if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
        threadpool_batch = ggml_threadpool_new(&tpp_batch);
        if (!threadpool_batch) {
            LOG_ERR("%s: batch threadpool create failed : n_threads %d\n", __func__, tpp_batch.n_threads);
            return_value.result = -1;
            return return_value;
        }

        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return_value.result = -2;
        return return_value;
    }
    llama_attach_threadpool(ctx, threadpool, threadpool_batch);


    // should we add the bos?
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }
    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);


    // tokenize the prompt (`embd_inp` in llamacpp samples)
    std::vector<llama_token> prompt_tokens;
    if (!params.prompt.empty()) {
        LOG_DBG("tokenize the prompt\n");
        prompt_tokens = ::common_tokenize(ctx, params.prompt, add_bos, true);
        LOG_DBG("prompt: \"%s\"\n", params.prompt.c_str());
        LOG_DBG("tokens: %s\n", string_from(ctx, prompt_tokens).c_str());
    }
    if (prompt_tokens.empty()) {
        // The model needs something to start with, so if we're empty
        // add the bos if the model supports it. Otherwise we start
        // with a newline character, but this should be considered a 
        // fallback and should be fixed in the calling client code.
        if (add_bos) {
            prompt_tokens.push_back(llama_vocab_bos(vocab));
            LOG_WRN("%s: prompt_tokens was considered empty and bos was added: %s\n", 
                __func__, string_from(ctx, prompt_tokens).c_str());
        } else {
            LOG_ERR("%s: error: input is empty and bos isn't supported so a newline will be used!\n", __func__);
            prompt_tokens = ::common_tokenize(ctx, "\n", add_bos, true);
            if (prompt_tokens.empty()) {
                LOG_ERR("%s: error: input is empty and failed to tokenize newline!\n", __func__);
                return_value.result = -3;
                return return_value;
            }
        }
    }

    const int prompt_token_limit = n_ctx - 4;
    if ((int) prompt_tokens.size() > prompt_token_limit) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", 
            __func__, (int) prompt_tokens.size(), prompt_token_limit);
        return_value.result = -4;
        return return_value;
    }

#ifdef WOOLY_DEBUG
        LOG_DBG("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_DBG("%s: number of tokens in prompt = %zu\n", __func__, prompt_tokens.size());
        for (int i = 0; i < (int) prompt_tokens.size(); i++) {
            LOG_DBG("%6d -> '%s'\n", prompt_tokens[i], llama_token_to_piece(ctx, prompt_tokens[i]).c_str());
        }
        LOG_DBG("\n");
#endif


    // setup the sampler to be used while ingesting the prompt
    common_sampler* smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n",  __func__);
        return_value.result = -5;
        return return_value;
    }
    LOG_INF("sampling seed: %u\n", common_sampler_get_seed(smpl));
    LOG_INF("sampling params: \n%s\n", params.sampling.print().c_str());
    LOG_INF("sampler chain: \n%s\n", common_sampler_print(smpl).c_str());
    LOG_INF("batch size n_batch = %d\n",params.n_batch);
    LOG_INF("\n\n");

    // NOTE: Might need extra work here for models with encoders

    // Ingest the prompt
    int32_t n_consumed = 0;
    while ((int)prompt_tokens.size() > n_consumed) {
        // push the prompt in the sampling context in order to apply repetition penalties later
        // for the prompt, we don't apply grammar rules
        common_sampler_accept(smpl, prompt_tokens[n_consumed], /* accept_grammar= */ false);
        ++n_consumed;
    }

    n_consumed = 0;
    for (int i = 0; i < (int)prompt_tokens.size(); i += params.n_batch) {
        int n_eval = (int)prompt_tokens.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }
        if (llama_decode(ctx, llama_batch_get_one(&prompt_tokens[i], n_eval))) {
            LOG_ERR("%s : failed to eval batch of %d at offset %d\n", __func__, n_eval, i);
            return_value.result = -6;
            return return_value;
        }
        n_consumed += n_eval;
        LOG_DBG("%s: prompt tokens consumed so far = %d / %zu\n", __func__, n_consumed, prompt_tokens.size());
    }

    return_value.result = n_consumed;
    return_value.gpt_sampler = (wooly_sampler_t*) smpl;
    return return_value;
}

int32_t
wooly_sample_next(
    wooly_llama_context_t* llama_context_ptr, 
    wooly_sampler_t* sampler_ptr) 
{
    llama_context *ctx = (llama_context *)llama_context_ptr; 
    common_sampler *smpl = (common_sampler *)sampler_ptr;

    const llama_token id = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, id, true);
    //LOG_DBG("last: %s\n", common_token_to_piece(ctx, id, true).c_str());

    return id;
}

int32_t
wooly_process_next_token(
    wooly_llama_context_t* llama_context_ptr, 
    int32_t next_token)
{
    llama_context *ctx = (llama_context *)llama_context_ptr; 
    if (llama_decode(ctx, llama_batch_get_one(&next_token, 1))) {
        LOG_ERR("%s : failed to evaluate the next token\n", __func__);
        return -1;
    }

    return 0;
}

int32_t
wooly_check_eog_and_antiprompt(
    wooly_gpt_params simple_params, 
    wooly_llama_context_t* llama_context_ptr, 
    wooly_llama_model_t* llama_model_ptr, 
    wooly_sampler_t* sampler_ptr) 
{
    llama_model *model = (llama_model *)llama_model_ptr; 
    llama_context *ctx = (llama_context *)llama_context_ptr; 
    common_sampler *smpl = (common_sampler *)sampler_ptr;
 
    // first, we check against the model's end of generation tokens
    const llama_vocab *vocab = llama_model_get_vocab(model);
    if (llama_vocab_is_eog(vocab, common_sampler_last(smpl))) {
        return 1;
    }

    // then we check against our antiprompts if supplied
    if (simple_params.antiprompt_count > 0 && simple_params.antiprompts != NULL) {
        const int n_prev = 32;
        const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

        bool is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the output.
        // If we're not running interactively, the reverse prompt might be tokenized with some following characters
        // so we'll compensate for that by widening the search window a bit.
        for (int i=0; i<simple_params.antiprompt_count; ++i) {
            std::string antiprompt(simple_params.antiprompts[i]);
            size_t extra_padding = 2;
            size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                : 0;

            if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                is_antiprompt = true;
                break;
            }
        }

        // tokenized antiprompts
        std::vector<std::vector<llama_token>> antiprompt_ids;
        antiprompt_ids.reserve(simple_params.antiprompt_count);
        for (int i=0; i<simple_params.antiprompt_count; ++i) {
            std::string antiprompt(simple_params.antiprompts[i]);
            antiprompt_ids.emplace_back(::common_tokenize(ctx, antiprompt, false, true));
        }

        // and this time check check for reverse prompt using special tokens
        llama_token last_token = common_sampler_last(smpl);
        for (std::vector<llama_token> ids : antiprompt_ids) {
            if (ids.size() == 1 && last_token == ids[0]) {
                is_antiprompt = true;
                break;
            }
        }

        if (is_antiprompt) {
            LOG_DBG("%s: found antiprompt: %s\n", __func__, last_output.c_str());
            return 2;
        }
    }

    return 0;
}

wooly_prompt_cache_t*
wooly_freeze_prediction_state(
    wooly_gpt_params        simple_params,
    wooly_llama_context_t*  llama_context_ptr,
    wooly_llama_model_t*    llama_model_ptr,
    int32_t *predicted_tokens,
    int64_t predicted_token_count)
{
    llama_model *model = (llama_model *)llama_model_ptr; 
    llama_context *ctx = (llama_context *)llama_context_ptr; 
    
    // must be passed the prompt use to freeze the current state
    if (strlen(simple_params.prompt) <= 0) {
        return NULL;
    }

    // tokenize the prompt
    const llama_vocab *vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_vocab_get_add_eos(vocab));
    }

    // tokenize the prompt
    std::vector<llama_token> prompt_tokens = ::common_tokenize(ctx, simple_params.prompt, add_bos, true);
    
    llama_predict_prompt_cache* prompt_cache = new llama_predict_prompt_cache;
    llama_synchronize(ctx);

    const size_t state_size = llama_state_get_size(ctx);
    prompt_cache->last_processed_prompt_state = new uint8_t[state_size];
    prompt_cache->last_processed_prompt_state_size = state_size;

    llama_state_get_data(ctx, prompt_cache->last_processed_prompt_state, state_size);
    prompt_cache->last_prompt = simple_params.prompt;
    prompt_cache->processed_prompt_tokens.insert(prompt_cache->processed_prompt_tokens.end(), prompt_tokens.begin(),prompt_tokens.end());

    // if we're freezing a state that includes predicted tokens, then include those too.
    if (predicted_tokens != NULL && predicted_token_count > 0) {
        std::vector<llama_token> predictions(predicted_tokens, predicted_tokens + predicted_token_count);
        prompt_cache->processed_prompt_tokens.insert(prompt_cache->processed_prompt_tokens.end(), predictions.begin(),predictions.end());
    }
    return (wooly_prompt_cache_t*) prompt_cache;
}

wooly_process_prompt_results
wooly_defrost_prediction_state(
    wooly_gpt_params        simple_params,
    wooly_llama_context_t*  llama_context_ptr, 
    wooly_llama_model_t*    llama_model_ptr, 
    wooly_prompt_cache_t*   prompt_cache_ptr)
{
    llama_model *model = (llama_model *)llama_model_ptr; 
    llama_context *ctx = (llama_context *)llama_context_ptr; 
    llama_predict_prompt_cache* prompt_cache = (llama_predict_prompt_cache *)prompt_cache_ptr; 
    wooly_process_prompt_results return_value;

    // build up a new sampler for the parameters provided
    common_params params;
    fill_params_from_simple(&simple_params, &params);
    common_sampler* smpl = common_sampler_init(model, params.sampling);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n",  __func__);
        return_value.result = -1;
        return return_value;
    }

    // feed the prompt back through the sampler to reset the sampler state
    common_sampler_reset(smpl);
    for (int i=0; i<prompt_cache->processed_prompt_tokens.size(); ++i) {
        common_sampler_accept(smpl, prompt_cache->processed_prompt_tokens[i], /* accept_grammar= */ false);
    }

    llama_synchronize(ctx);
    llama_state_set_data(ctx, prompt_cache->last_processed_prompt_state, prompt_cache->last_processed_prompt_state_size);

    return_value.gpt_sampler = (wooly_sampler_t*) smpl;
    return_value.result = (int32_t) prompt_cache->processed_prompt_tokens.size();
    return return_value;
}


wooly_predict_result 
wooly_predict(
    wooly_gpt_params        simple_params, 
    wooly_llama_context_t*  llama_context_ptr, 
    wooly_llama_model_t*    llama_model_ptr, 
    bool                    include_specials, 
    char*                   out_result, 
    int64_t                 out_result_size,
    wooly_prompt_cache_t*   prompt_cache_ptr, 
    wooly_token_update_callback token_cb) 
{
    llama_context *ctx = (llama_context *)llama_context_ptr; 

    wooly_predict_result results;
    int64_t t_start_us = ggml_time_us();

    // keeps track of how many prompt tokens were processed.
    // note: this will remain zero if the prompt cache is used.
    int32_t prompt_token_count = 0;

    bool cache_used = false;
    wooly_process_prompt_results prompt_results;
    wooly_prompt_cache_t* prompt_cache = prompt_cache_ptr;
    if (prompt_cache != NULL) {
        // check to see if the prompts are equal and if they are, restore the cached state.
        // if it is not a match, then
        llama_predict_prompt_cache* prev_cache = (llama_predict_prompt_cache*)prompt_cache;
        if (prev_cache->last_prompt == simple_params.prompt) {
            // restore prediction state from the prompt cache
            prompt_results = wooly_defrost_prediction_state(
                simple_params, 
                llama_context_ptr, 
                llama_model_ptr,
                prompt_cache);
            cache_used = true;
        } else {
            // don't use the cache, but since this was passed in by the client,
            // the client is responsible for freeing it.
            prompt_cache = NULL;
            cache_used = false;
        }
    } 
    if (!cache_used) {
        // get the prompt ingested into the context and pull the sampler
        // used in the process so that repeat penalties and such are
        // accounted for.
        prompt_results = wooly_process_prompt(
            simple_params, 
            llama_context_ptr, 
            llama_model_ptr);
    

        // if the caller wishes to cache the prompt, freeze the prediction state here
        if (simple_params.prompt_cache_all) {
            prompt_cache = wooly_freeze_prediction_state(
                simple_params, 
                llama_context_ptr, 
                llama_model_ptr,
                NULL,
                0);
        }
        
        prompt_token_count = prompt_results.result;
    }

    int64_t t_p_eval_end_us = ggml_time_us();

    // pull the sampler from the prompt processing
    wooly_sampler_t* sampler_ptr = prompt_results.gpt_sampler;


    // calculate how many tokens to predict; passing -1 means we use the maximum
    // amount remaining in the context.
    int32_t num_to_predict = simple_params.n_predict != -1  
        ? simple_params.n_predict 
        : llama_n_ctx(ctx) - prompt_results.result - 4;


    // zero out our prediction token array
    int32_t *predicted_tokens = static_cast<int32_t *>(calloc(num_to_predict, sizeof(int32_t)));


    // run a prediction loop
    int32_t predicted = 0;
    while (predicted < num_to_predict) {
        predicted_tokens[predicted] = wooly_sample_next(llama_context_ptr, sampler_ptr);
        
        // call the token callback with the newly predicted token
        if (token_cb != NULL) {
            auto token_str = common_token_to_piece(ctx, predicted_tokens[predicted], include_specials);
            if (!token_cb(token_str.c_str())) {
                break;
            }
        }

        // do all the antiprompt testing and eog testing
        int32_t eog = wooly_check_eog_and_antiprompt(simple_params, llama_context_ptr, llama_model_ptr, sampler_ptr);
        if (eog > 0) {
            break;
        }

        // calculate the next logits (expensive compute)
        int32_t success = wooly_process_next_token(
            llama_context_ptr, 
            predicted_tokens[predicted]);

        predicted++;
    }

    // convert our predicted tokens to a string and print the result
    int64_t pred_str_len = wooly_llama_detokenize(
        llama_context_ptr, 
        include_specials, 
        predicted_tokens, 
        predicted, 
        out_result, 
        out_result_size);


    int64_t t_eval_end_us = ggml_time_us();

    // update the prediction statistics
    int64_t t_end_us = ggml_time_us();
    results.t_end_ms =  1e-3 * t_end_us;
    results.t_start_ms =  1e-3 * t_start_us;
    results.t_p_eval_ms = 1e-3 * (t_p_eval_end_us - t_start_us);
    results.t_eval_ms = 1e-3 * (t_eval_end_us - t_p_eval_end_us);
    results.n_p_eval = prompt_token_count;
    results.n_eval = predicted;
    results.prompt_cache = prompt_cache;
    results.result = 0;


    // free our allocated resources
    wooly_free_sampler(sampler_ptr);
    free(predicted_tokens);


    // negative values get returned if the buffer isn't big enough, so return that
    // as the error code.
    if (pred_str_len < 0) {
        results.result = pred_str_len;
        return results;
    }
    
    return results;
}

void 
wooly_free_prompt_cache(wooly_prompt_cache_t* prompt_cache_ptr)
{
    if (prompt_cache_ptr != nullptr) {
        llama_predict_prompt_cache *prompt_cache_data = (llama_predict_prompt_cache *) prompt_cache_ptr;
        delete[] prompt_cache_data->last_processed_prompt_state;
        delete prompt_cache_data;
    }
}


LLAMA_API wooly_gpt_params 
wooly_new_gpt_params()
{
    wooly_gpt_params output;
    common_params prototype;

    // copy default values from the prototype onto the output structure

    output.prompt = nullptr;
    output.antiprompts = nullptr;
    output.antiprompt_count = 0;
    output.seed = prototype.sampling.seed;
    output.n_threads = prototype.cpuparams.n_threads;
    output.n_threads_batch = prototype.cpuparams_batch.n_threads;
    output.n_predict = prototype.n_predict;
    output.n_ctx = prototype.n_ctx;
    output.n_batch = prototype.n_batch;
    output.n_gpu_layers = prototype.n_gpu_layers;
    output.split_mode = prototype.split_mode;
    output.main_gpu = prototype.main_gpu;
    memcpy(&output.tensor_split, &prototype.tensor_split, sizeof(float) * 128);
    output.grp_attn_n = prototype.grp_attn_n;
    output.grp_attn_w = prototype.grp_attn_w;
    output.rope_freq_base = prototype.rope_freq_base;
    output.rope_freq_scale = prototype.rope_freq_scale;
    output.yarn_ext_factor = prototype.yarn_ext_factor;
    output.yarn_attn_factor = prototype.yarn_attn_factor;
    output.yarn_beta_fast = prototype.yarn_beta_fast;
    output.yarn_beta_slow = prototype.yarn_beta_slow;
    output.yarn_orig_ctx = prototype.yarn_orig_ctx;
    output.rope_scaling_type = prototype.rope_scaling_type;
    output.prompt_cache_all = prototype.prompt_cache_all;
    output.ignore_eos = prototype.sampling.ignore_eos;
    output.flash_attn = prototype.flash_attn;

    output.embedding = prototype.embedding;
    output.embd_normalize = prototype.embd_normalize;

    output.top_k = prototype.sampling.top_k;
    output.top_p = prototype.sampling.top_p;
    output.min_p = prototype.sampling.min_p;
    output.xtc_probability = prototype.sampling.xtc_probability;
    output.xtc_threshold = prototype.sampling.xtc_threshold;
    output.typical_p = prototype.sampling.typ_p;
    output.temp = prototype.sampling.temp;
    output.dynatemp_range = prototype.sampling.dynatemp_range;
    output.dynatemp_exponent = prototype.sampling.dynatemp_exponent;
    output.penalty_last_n = prototype.sampling.penalty_last_n;
    output.penalty_repeat = prototype.sampling.penalty_repeat;
    output.penalty_freq = prototype.sampling.penalty_freq;
    output.penalty_present = prototype.sampling.penalty_present;
    output.dry_multiplier = prototype.sampling.dry_multiplier;
    output.dry_base = prototype.sampling.dry_base;
    output.dry_allowed_length = prototype.sampling.dry_allowed_length;
    output.dry_penalty_last_n = prototype.sampling.dry_penalty_last_n;
    output.dry_sequence_breakers = nullptr;
    output.dry_sequence_breakers_count = 0;
    output.mirostat = prototype.sampling.mirostat;
    output.mirostat_tau = prototype.sampling.mirostat_tau;
    output.mirostat_eta = prototype.sampling.mirostat_eta;
    output.grammar = nullptr;

    return output;
}

void 
fill_params_from_simple(
    wooly_gpt_params *simple, 
    common_params *output)
{
    if (simple->prompt != nullptr) {
        output->prompt = simple->prompt;
    }
    if (simple->antiprompt_count > 0)
    {
        output->antiprompt = create_vector(simple->antiprompts, simple->antiprompt_count);
    }

    output->sampling.seed = simple->seed;
    output->cpuparams.n_threads = simple->n_threads;
    if (output->cpuparams.n_threads < 1) {
        output->cpuparams.n_threads = cpu_get_num_math();
    }
    output->cpuparams_batch.n_threads = simple->n_threads_batch;
    if (output->cpuparams_batch.n_threads < 1) {
        output->cpuparams_batch.n_threads = output->cpuparams.n_threads;
    }
    output->n_predict = simple->n_predict;
    output->n_ctx = simple->n_ctx;
    output->n_batch = simple->n_batch;
    output->n_gpu_layers = simple->n_gpu_layers;
    output->split_mode = (llama_split_mode) simple->split_mode;
    output->main_gpu = simple->main_gpu;
    memcpy(&output->tensor_split, &simple->tensor_split, sizeof(float) * 128);
    output->grp_attn_n = simple->grp_attn_n;
    output->grp_attn_w = simple->grp_attn_w;
    output->rope_freq_base = simple->rope_freq_base;
    output->rope_freq_scale = simple->rope_freq_scale;
    output->yarn_ext_factor = simple->yarn_ext_factor;
    output->yarn_attn_factor = simple->yarn_attn_factor;
    output->yarn_beta_fast = simple->yarn_beta_fast;
    output->yarn_beta_slow = simple->yarn_beta_slow;
    output->yarn_orig_ctx = simple->yarn_orig_ctx;
    output->rope_scaling_type = (llama_rope_scaling_type) simple->rope_scaling_type;
    output->prompt_cache_all = simple->prompt_cache_all;
    output->sampling.ignore_eos = simple->ignore_eos;
    output->flash_attn = simple->flash_attn;

    output->embedding = simple->embedding;
    output->embd_normalize = simple->embd_normalize;

    output->sampling.top_k = simple->top_k;
    output->sampling.top_p = simple->top_p;
    output->sampling.min_p = simple->min_p;
    output->sampling.xtc_probability = simple->xtc_probability;
    output->sampling.xtc_threshold = simple->xtc_threshold;
    output->sampling.typ_p = simple->typical_p;
    output->sampling.temp = simple->temp;
    output->sampling.dynatemp_range = simple->dynatemp_range;
    output->sampling.dynatemp_exponent = simple->dynatemp_exponent;
    output->sampling.penalty_last_n = simple->penalty_last_n;
    output->sampling.penalty_repeat = simple->penalty_repeat;
    output->sampling.penalty_freq = simple->penalty_freq;
    output->sampling.penalty_present = simple->penalty_present;
    output->sampling.dry_multiplier = simple->dry_multiplier;
    output->sampling.dry_base = simple->dry_base;
    output->sampling.dry_allowed_length = simple->dry_allowed_length;
    output->sampling.dry_penalty_last_n = simple->dry_penalty_last_n;
    if (simple->dry_sequence_breakers_count > 0)
    {
        output->sampling.dry_sequence_breakers = create_vector(simple->dry_sequence_breakers, simple->dry_sequence_breakers_count);
    }
    output->sampling.mirostat = simple->mirostat;
    output->sampling.mirostat_tau = simple->mirostat_tau;
    output->sampling.mirostat_eta = simple->mirostat_eta;
    if (simple->grammar != nullptr) {
        output->sampling.grammar = simple->grammar;
    }
}

wooly_llama_model_params
wooly_get_default_llama_model_params()
{
    llama_model_params defaults = llama_model_default_params();
    
    wooly_llama_model_params output;
    output.n_gpu_layers = defaults.n_gpu_layers;
    output.split_mode = (int32_t) defaults.split_mode;
    output.main_gpu = defaults.main_gpu;
    output.tensor_split = defaults.tensor_split; // shallow copy
    output.vocab_only = defaults.vocab_only;
    output.use_mmap = defaults.use_mmap;
    output.use_mlock = defaults.use_mlock;
    output.check_tensors = defaults.check_tensors;
    return output;
}

llama_model_params
conv_wooly_to_llama_model_params(wooly_llama_model_params wooly_params)
{
    llama_model_params params = llama_model_default_params();
    
    params.n_gpu_layers = wooly_params.n_gpu_layers;
    params.split_mode = (llama_split_mode) wooly_params.split_mode;
    params.main_gpu = wooly_params.main_gpu;
    params.tensor_split = wooly_params.tensor_split; // shallow copy
    params.vocab_only = wooly_params.vocab_only;
    params.use_mmap = wooly_params.use_mmap;
    params.use_mlock = wooly_params.use_mlock;
    params.check_tensors = wooly_params.check_tensors;
    
    return params;
}


LLAMA_API wooly_llama_context_params
wooly_get_default_llama_context_params()
{
    llama_context_params defaults = llama_context_default_params();

    wooly_llama_context_params output;
    output.n_ctx = defaults.n_ctx;
    output.n_batch = defaults.n_batch;
    output.n_ubatch = defaults.n_ubatch;
    output.n_seq_max = defaults.n_seq_max;
    output.n_threads = defaults.n_threads;
    output.n_threads_batch = defaults.n_threads_batch;
    output.rope_scaling_type = (int32_t) defaults.rope_scaling_type;
    output.pooling_type = (int32_t) defaults.pooling_type;
    output.rope_freq_base = defaults.rope_freq_base;
    output.rope_freq_scale = defaults.rope_freq_scale;
    output.yarn_ext_factor = defaults.yarn_ext_factor;
    output.yarn_attn_factor = defaults.yarn_attn_factor;
    output.yarn_beta_fast = defaults.yarn_beta_fast;
    output.yarn_beta_slow = defaults.yarn_beta_slow;
    output.yarn_attn_factor = defaults.yarn_attn_factor;
    output.defrag_thold = defaults.defrag_thold;
    output.logits_all = defaults.logits_all;
    output.embeddings = defaults.embeddings;
    output.offload_kqv = defaults.offload_kqv;
    output.flash_attn = defaults.flash_attn;

    return output;
}

llama_context_params
conv_wooly_to_llama_context_params(wooly_llama_context_params wooly_params)
{
    llama_context_params params = llama_context_default_params();
    params.no_perf = false;

    params.n_ctx = wooly_params.n_ctx;
    params.n_batch = wooly_params.n_batch;
    params.n_ubatch = wooly_params.n_ubatch;
    params.n_seq_max = wooly_params.n_seq_max;
    params.n_threads = wooly_params.n_threads;
    if (params.n_threads < 1) {
        params.n_threads = cpu_get_num_math();
    }
    params.n_threads_batch = wooly_params.n_threads_batch;
    if (params.n_threads_batch < 1) {
        params.n_threads_batch =  params.n_threads;
    }
    params.rope_scaling_type = (enum llama_rope_scaling_type) wooly_params.rope_scaling_type;
    params.pooling_type = (enum llama_pooling_type) wooly_params.pooling_type;
    params.rope_freq_base = wooly_params.rope_freq_base;
    params.rope_freq_scale = wooly_params.rope_freq_scale;
    params.yarn_ext_factor = wooly_params.yarn_ext_factor;
    params.yarn_attn_factor = wooly_params.yarn_attn_factor;
    params.yarn_beta_fast = wooly_params.yarn_beta_fast;
    params.yarn_beta_slow = wooly_params.yarn_beta_slow;
    params.yarn_attn_factor = wooly_params.yarn_attn_factor;
    params.defrag_thold = wooly_params.defrag_thold;
    params.logits_all = wooly_params.logits_all;
    params.embeddings = wooly_params.embeddings;
    params.offload_kqv = wooly_params.offload_kqv;
    params.flash_attn = wooly_params.flash_attn;

    return params;
}

int32_t 
wooly_llama_n_embd(wooly_llama_model_t *llama_model_ptr) 
{
    return llama_model_n_embd((const llama_model *)llama_model_ptr);
}

int64_t
wooly_llama_tokenize(
    wooly_llama_context_t*llama_ctx_ptr, 
    const char* text,
    bool add_special,
    bool parse_special,
    int32_t* out_tokens,
    int64_t out_tokens_size)
{
    auto tokens = common_tokenize(
        (const llama_context *)llama_ctx_ptr, 
        text, 
        add_special, 
        parse_special);

    if (out_tokens != NULL) {
        // clip the maximum copy size to the output buffer size
        size_t copy_size = std::min(out_tokens_size, static_cast<int64_t>(tokens.size()));
        std::copy(tokens.begin(), tokens.begin() + copy_size, out_tokens);
        return copy_size;
    } else {
        return tokens.size();
    }
}

int64_t 
wooly_llama_detokenize(
    wooly_llama_context_t* llama_context_ptr, 
    bool render_specials, 
    int32_t* tokens,
    int64_t tokens_size,
    char *out_result, 
    int64_t out_result_size)
{
    // build the STL vector out of the input buffer for the tokens
    std::vector<int32_t> input_tokens(tokens, tokens + tokens_size);
        
    // render the tokens out to the string
    auto string = common_detokenize(
        (llama_context*)llama_context_ptr, 
        input_tokens, 
        render_specials);

    // make sure we have enough space in the output buffer
    if (string.length() + 1 > (size_t) out_result_size) {
        return -((int64_t) string.length() + 1);
    }

    // copy the result and then return the length of the string
    std::strcpy(out_result, string.c_str());
    return string.length();
}

bool
wooly_has_chat_template(
    const wooly_llama_model_t*     llama_model_ptr)
{
    auto str = llama_model_chat_template((const llama_model *)llama_model_ptr, /* name */ nullptr);
    if (str) {
        return true;
    }

    return false;
}

int64_t
wooly_apply_chat_template(
    const wooly_llama_model_t*      llama_model_ptr,
    const char*                     chat_template,
    bool                            include_assistant,
    const wooly_chat_message*       chat_messages,
    int64_t                         chat_message_count,
    char *                          out_result, 
    int64_t                         out_result_size)
{
    // try to pull a chat template from the model file
    auto tmpl = llama_model_chat_template((const llama_model *)llama_model_ptr, chat_template);
    
    // if getting the overrided chat template by name fails, then we'll just use it like a standard
    // 'common name' for a supported template format.
    if (tmpl == NULL) {
        if (chat_template == NULL) {
            tmpl = "chatml";
        } else {
            tmpl = chat_template;
        }
    }

    // build up the native type it expects.
    llama_chat_message* llama_chat_msgs = new llama_chat_message[chat_message_count];
    for (int64_t i = 0; i < chat_message_count; ++i) {
        llama_chat_msgs[i].role = chat_messages[i].role;
        llama_chat_msgs[i].content = chat_messages[i].content;
    }

    // do the actual application of the template specified.
    int actual_len = llama_chat_apply_template(tmpl, llama_chat_msgs, chat_message_count, include_assistant, out_result, out_result_size);

    // done with our temporary conversion
    delete[] llama_chat_msgs;

    return actual_len;
}

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

int64_t
wooly_llama_make_embeddings(
    wooly_llama_model_t *llama_model_ptr,
    wooly_llama_context_t *llama_context_ptr,
    int32_t batch_size,
    int32_t pooling_type,
    int32_t embd_normalize,
    int64_t token_array_count,
    int32_t** token_arrays,
    int64_t* token_array_sizes,
    float* output_embeddings,
    int64_t output_embeddings_size)
{
    // count number of embeddings; if no pooling, then each token has its own
    // embedding vector, otherwise all of the embeddings for a given prompt are processed and
    // reduced down to one embedding vector.
    // n_embd_count ends up being the total number of embedding vectors needed for the output.
    int64_t n_embd_count = 0;
    if (pooling_type == LLAMA_POOLING_TYPE_NONE) {
        for (int k = 0; k < token_array_count; k++) {
            n_embd_count += token_array_sizes[k];
        }
    } else {
        n_embd_count = token_array_count;
    }

    // allocate output
    const int n_embd = wooly_llama_n_embd(llama_model_ptr);
    if (output_embeddings_size < n_embd_count * n_embd) {
        return -(n_embd_count * n_embd); // output buffer is too small
    }

    // break into batches
    struct llama_batch batch = llama_batch_init(batch_size, 0, 1);
    int e = 0; // number of embeddings already stored
    int s = 0; // number of prompts in current batch
    for (int k = 0; k < token_array_count; k++) {
        std::vector<int32_t> inp(token_arrays[k], token_arrays[k] + token_array_sizes[k]);
        const uint64_t n_toks = inp.size();

        // encode if at capacity
        if (batch.n_tokens + n_toks > batch_size) {
            float * out = output_embeddings + e * n_embd;
            batch_decode_embeddings((llama_context *)llama_context_ptr, batch, out, s, n_embd, embd_normalize);
            e += pooling_type == LLAMA_POOLING_TYPE_NONE ? batch.n_tokens : s;
            s = 0;
            common_batch_clear(batch);
        }

        // add to batch
        batch_add_seq(batch, inp, s);
        s += 1;
    }

    // final batch - after this our embeddings vector should be fully populated
    float * out = output_embeddings + e * n_embd;
    batch_decode_embeddings((llama_context *)llama_context_ptr, batch, out, s, n_embd, embd_normalize);

    return 0;
}
