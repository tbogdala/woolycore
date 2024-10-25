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

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
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
        model = llama_load_model_from_file(fname, model_params);
        if (model == NULL) {
            return res;
        }
	    lctx = llama_new_context_with_model(model, context_params);
        if (lctx == NULL) {
            llama_free_model(model);
            return res;
        }
    }
    catch (std::runtime_error &e)
    {
        LOG_ERR("failed %s", e.what());
        llama_free(lctx);
        llama_free_model(model);
        return res;
    }

    {
        LOG_WRN("warming up the model with an empty run\n");

         std::vector<llama_token> tmp;
        llama_token bos = llama_token_bos(model);
        llama_token eos = llama_token_eos(model);
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

    res.ctx = lctx;
    res.model = model;
    res.context_length = llama_n_ctx(lctx);
    return res;
}

void 
wooly_free_model(
    void *llama_context_ptr, 
    void *llama_model_ptr)
{
    llama_context *ctx = (llama_context *) llama_context_ptr; 
    llama_model *model = (llama_model *) llama_model_ptr;
    if (model != NULL) {
        llama_free_model(model);
    }
    if (ctx != NULL) {
        llama_free(ctx);
    }
}

void 
wooly_free_sampler(
    void *sampler_ptr)
{
    if (sampler_ptr != NULL) {
        common_sampler *smpl = static_cast<common_sampler *>(sampler_ptr);
        common_sampler_free(smpl);
    }
}

wooly_process_prompt_results 
wooly_process_prompt(
    wooly_gpt_params simple_params, 
    void *llama_context_ptr, 
    void *llama_model_ptr) 
{
    wooly_process_prompt_results return_value;
    llama_context *ctx = static_cast<llama_context *>(llama_context_ptr); 
    llama_model *model = static_cast<llama_model *>(llama_model_ptr);

    common_params params;
    fill_params_from_simple(&simple_params, &params);


    // Do a basic set of warnings based on incoming parameters
    const int n_ctx_train = llama_n_ctx_train(model);
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
    const bool add_bos = llama_add_bos_token(model);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_add_eos_token(model));
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
            prompt_tokens.push_back(llama_token_bos(model));
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
    common_sampler* smpl = common_sampler_init(model, params.sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n",  __func__);
        return_value.result = -5;
        return return_value;
    }
    LOG_INF("sampling seed: %u\n", common_sampler_get_seed(smpl));
    LOG_INF("sampling params: \n%s\n", params.sparams.print().c_str());
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
    return_value.gpt_sampler = smpl;
    return return_value;
}

int32_t
wooly_sample_next(
    void *llama_context_ptr, 
    void *sampler_ptr) 
{
    llama_context *ctx = static_cast<llama_context *>(llama_context_ptr); 
    common_sampler *smpl = static_cast<common_sampler *>(sampler_ptr);

    const llama_token id = common_sampler_sample(smpl, ctx, -1);
    common_sampler_accept(smpl, id, true);
    //LOG_DBG("last: %s\n", common_token_to_piece(ctx, id, true).c_str());

    return id;
}

int32_t
wooly_process_next_token(
    void *llama_context_ptr, 
    int32_t next_token)
{
    llama_context *ctx = static_cast<llama_context *>(llama_context_ptr); 
    if (llama_decode(ctx, llama_batch_get_one(&next_token, 1))) {
        LOG_ERR("%s : failed to evaluate the next token\n", __func__);
        return -1;
    }

    return 0;
}

int32_t
wooly_check_eog_and_antiprompt(
    wooly_gpt_params simple_params, 
    void *llama_context_ptr, 
    void *llama_model_ptr, 
    void *sampler_ptr) 
{
    llama_context *ctx = static_cast<llama_context *>(llama_context_ptr); 
    llama_model *model = static_cast<llama_model *>(llama_model_ptr); 
    common_sampler *smpl = static_cast<common_sampler *>(sampler_ptr);
 
    // first, we check against the model's end of generation tokens
    if (llama_token_is_eog(model, common_sampler_last(smpl))) {
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

void*
wooly_freeze_prediction_state(
    wooly_gpt_params simple_params,
    void *llama_context_ptr,
    void *llama_model_ptr,
    int32_t *predicted_tokens,
    int64_t predicted_token_count)
{
    llama_context *ctx = static_cast<llama_context *>(llama_context_ptr); 
    llama_model *model = static_cast<llama_model *>(llama_model_ptr); 
    
    // must be passed the prompt use to freeze the current state
    if (strlen(simple_params.prompt) <= 0) {
        return NULL;
    }

    // tokenize the prompt
    const bool add_bos = llama_add_bos_token(model);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_add_eos_token(model));
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
    return prompt_cache;
}

wooly_process_prompt_results
wooly_defrost_prediction_state(
    wooly_gpt_params simple_params,
    void *llama_context_ptr, 
    void *llama_model_ptr, 
    void *prompt_cache_ptr)
{
    llama_context *ctx = static_cast<llama_context *>(llama_context_ptr); 
    llama_model *model = static_cast<llama_model *>(llama_model_ptr); 
    llama_predict_prompt_cache* prompt_cache = static_cast<llama_predict_prompt_cache *>(prompt_cache_ptr); 
    wooly_process_prompt_results return_value;

    // build up a new sampler for the parameters provided
    common_params params;
    fill_params_from_simple(&simple_params, &params);
    common_sampler* smpl = common_sampler_init(model, params.sparams);
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

    return_value.gpt_sampler = smpl;
    return_value.result = (int32_t) prompt_cache->processed_prompt_tokens.size();
    return return_value;
}


wooly_predict_result 
wooly_predict(
    wooly_gpt_params simple_params, 
    void *llama_context_ptr, 
    void *llama_model_ptr, 
    bool include_specials, 
    char *out_result, 
    int64_t out_result_size,
    void* prompt_cache_ptr, 
    wooly_token_update_callback token_cb) 
{
    llama_context *ctx = (llama_context *) llama_context_ptr; 
    llama_model *model = (llama_model *) llama_model_ptr;
    common_sampler *smpl = nullptr;
    common_params params;
    fill_params_from_simple(&simple_params, &params);

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
    
    llama_predict_prompt_cache *prompt_cache_data = (llama_predict_prompt_cache *) prompt_cache_ptr;
    wooly_predict_result return_value;
    return_value.n_eval = return_value.n_p_eval = 0;

    llama_kv_cache_clear(ctx);
    llama_perf_context_reset(ctx);
        
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
            return_value.result = 8;
            return return_value;
        }

        // Start the non-batch threadpool in the paused state
        tpp.paused = true;
    }

    struct ggml_threadpool * threadpool = ggml_threadpool_new(&tpp);
    if (!threadpool) {
        LOG_ERR("%s: threadpool create failed : n_threads %d\n", __func__, tpp.n_threads);
        return_value.result = 9;
        return return_value;
    }

    llama_attach_threadpool(ctx, threadpool, threadpool_batch);
    
    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);

    if (n_ctx > n_ctx_train) {
        LOG_WRN("%s: warning: model was trained on only %d context tokens (%d specified)\n", __func__, n_ctx_train, n_ctx);
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool resuse_last_prompt_data = false;
    if (prompt_cache_data != nullptr && params.prompt_cache_all) {
        // check to see if we're repeating the same prompt and reuse the stored prompt data if so.
        // if it's not a match, clear out the cached tokens.
        if (prompt_cache_data->last_prompt == params.prompt) {
            LOG_INF("Prompt match detected. Going to attempt to use last processed prompt token data and state.\n");
            resuse_last_prompt_data = true;
            llama_state_set_data(ctx, prompt_cache_data->last_processed_prompt_state, prompt_cache_data->last_processed_prompt_state_size);
        } else {
            // new prompt detected, so free the memory of the cached state
            if (prompt_cache_data->last_processed_prompt_state != nullptr) {
                delete[] prompt_cache_data->last_processed_prompt_state;
                prompt_cache_data->last_processed_prompt_state = nullptr;
                prompt_cache_data->last_processed_prompt_state_size = 0;
            }
            prompt_cache_data->processed_prompt_tokens.clear();
        }
    } else {
        // if we don't have a prompt cache object, create one
        prompt_cache_data = new llama_predict_prompt_cache;
        prompt_cache_data->last_processed_prompt_state = nullptr;
        prompt_cache_data->last_processed_prompt_state_size = 0;
    }
    // also copy the pointer of the prompt_cache_data to the result here now that it's for sure allocated
    return_value.prompt_cache = prompt_cache_data;

    std::string path_session = params.path_prompt_cache;
    std::vector<llama_token> session_tokens;

    if (!path_session.empty()) {
        LOG_INF("%s: attempting to load saved session from '%s'\n", __func__, path_session.c_str());
        if (!file_exists(path_session)) {
            LOG_INF("%s: session file does not exist, will create.\n", __func__);
        } else if (file_is_empty(path_session)) {
            LOG_INF("%s: The session file is empty. A new session will be initialized.\n", __func__);
        } else {
            // The file exists and is not empty
            session_tokens.resize(n_ctx);
            size_t n_token_count_out = 0;
            if (!llama_state_load_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.capacity(), &n_token_count_out)) {
                LOG_ERR("%s: failed to load session file '%s'\n", __func__, path_session.c_str());
                return_value.result = 1;
                return return_value;
            }
            session_tokens.resize(n_token_count_out);
            LOG_INF("%s: loaded a session with prompt size of %d tokens\n", __func__, (int)session_tokens.size());
        }
    }

    const bool add_bos = llama_add_bos_token(model);
    if (!llama_model_has_encoder(model)) {
        GGML_ASSERT(!llama_add_eos_token(model));
    }
    LOG_DBG("n_ctx: %d, add_bos: %d\n", n_ctx, add_bos);

    std::vector<llama_token> embd_inp;
    if (!resuse_last_prompt_data) {
        if (!params.prompt.empty() || session_tokens.empty()) {
            LOG_DBG("tokenize the prompt\n");
            embd_inp = ::common_tokenize(ctx, params.prompt, add_bos, true);
        } else {
            LOG_DBG("use session tokens\n");
            embd_inp = session_tokens;
        }
    }

    LOG_DBG("prompt: \"%s\"\n", params.prompt.c_str());
    LOG_DBG("tokens: %s\n", string_from(ctx, embd_inp).c_str());

    // Should not run without any tokens
    if (embd_inp.empty() && !resuse_last_prompt_data) {
        if (add_bos) {
            embd_inp.push_back(llama_token_bos(model));
            LOG_WRN("embd_inp was considered empty and bos was added: %s\n", string_from(ctx, embd_inp).c_str());
        } else {
            LOG_ERR("error: input is empty\n");
            return_value.result = 7;
            return return_value;
        }
    }

    if ((int) embd_inp.size() > n_ctx - 4) {
        LOG_ERR("%s: prompt is too long (%d tokens, max %d)\n", __func__, (int) embd_inp.size(), n_ctx - 4);
        return_value.result = 2;
        return return_value;
    }

    // debug message about similarity of saved session, if applicable
    size_t n_matching_session_tokens = 0;
    if (!session_tokens.empty()) {
        for (llama_token id : session_tokens) {
            if (n_matching_session_tokens >= embd_inp.size() || id != embd_inp[n_matching_session_tokens]) {
                break;
            }
            n_matching_session_tokens++;
        }
        if (params.prompt.empty() && n_matching_session_tokens == embd_inp.size()) {
            LOG_INF("%s: using full prompt from session file\n", __func__);
        } else if (n_matching_session_tokens >= embd_inp.size()) {
            LOG_INF("%s: session file has exact match for prompt!\n", __func__);
        } else if (n_matching_session_tokens < (embd_inp.size() / 2)) {
            LOG_WRN("%s: session file has low similarity to prompt (%zu / %zu tokens); will mostly be reevaluated\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        } else {
            LOG_INF("%s: session file matches %zu / %zu tokens of prompt\n",
                    __func__, n_matching_session_tokens, embd_inp.size());
        }

        // remove any "future" tokens that we might have inherited from the previous session
        llama_kv_cache_seq_rm(ctx, -1, (int32_t) n_matching_session_tokens, -1);
    }

    LOG_DBG("recalculate the cached logits (check): embd_inp.size() %zu, n_matching_session_tokens %zu, embd_inp.size() %zu, session_tokens.size() %zu\n",
        embd_inp.size(), n_matching_session_tokens, embd_inp.size(), session_tokens.size());

    // if we will use the cache for the full prompt without reaching the end of the cache, force
    // reevaluation of the last token to recalculate the cached logits
    if (!embd_inp.empty() && n_matching_session_tokens == embd_inp.size() && session_tokens.size() > embd_inp.size()) {
        LOG_DBG("recalculate the cached logits (do): session_tokens.resize( %zu )", embd_inp.size() - 1);

        session_tokens.resize(embd_inp.size() - 1);
    }

    // number of tokens to keep when resetting context
    if (params.n_keep < 0 || params.n_keep > (int)embd_inp.size()) {
        params.n_keep = (int)embd_inp.size();
    } else {
        params.n_keep += add_bos; // always keep the BOS token
    }

#ifdef WOOLY_DEBUG
        LOG_DBG("%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        LOG_DBG("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            LOG_DBG("%6d -> '%s'\n", embd_inp[i], llama_token_to_piece(ctx, embd_inp[i]).c_str());
        }

       if (params.n_keep > add_bos) {
            LOG_DBG("%s: static prompt based on n_keep: '", __func__);
            for (int i = 0; i < params.n_keep; i++) {
                LOG_CNT("%s", llama_token_to_piece(ctx, embd_inp[i]).c_str());
            }
            LOG_CNT("'\n");
        }
        LOG_DBG("\n");
#endif

    smpl = common_sampler_init(model, params.sparams);
    if (!smpl) {
        LOG_ERR("%s: failed to initialize sampling subsystem\n",  __func__);
        return_value.result = 10;
        return return_value;
    }

    LOG_INF("sampling seed: %u\n", common_sampler_get_seed(smpl));
    LOG_INF("sampling params: \n%s\n", params.sparams.print().c_str());
    LOG_INF("sampler chain: \n%s\n", common_sampler_print(smpl).c_str());
    LOG_INF("generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", n_ctx, params.n_batch, params.n_predict, params.n_keep);
    LOG_INF("\n\n");

    bool is_antiprompt        = false;
    bool need_to_save_session = !path_session.empty() && n_matching_session_tokens < embd_inp.size();

    int n_past             = 0;
    int n_remain           = params.n_predict;
    int n_consumed         = 0;
    int n_session_consumed = 0;

    std::vector<llama_token> embd;

    // tokenized antiprompts
    std::vector<std::vector<llama_token>> antiprompt_ids;

    antiprompt_ids.reserve(params.antiprompt.size());
    for (const std::string & antiprompt : params.antiprompt) {
        antiprompt_ids.emplace_back(::common_tokenize(ctx, antiprompt, false, true));
    }

    // our result to send back to woolycore bindings
    std::string res = "";
    bool need_to_save_state = true;

    // if we're reusing the prompt, clear out any input tokens to be processed
    // and set the tracking counter to the length of the saved prompt
    if (resuse_last_prompt_data) {
        embd_inp.clear();
        n_past = (int32_t) prompt_cache_data->processed_prompt_tokens.size();
        LOG_INF("%s: reusing prompt tokens; initializing n_consumed to %d\n",  __func__, n_consumed);
    } 
    else if (llama_model_has_encoder(model)) {
        int enc_input_size = (int) embd_inp.size();
        llama_token * enc_input_buf = embd_inp.data();

        if (llama_encode(ctx, llama_batch_get_one(enc_input_buf, enc_input_size))) {
            LOG_ERR("%s : failed to eval\n", __func__);
            return_value.result = 6;
            return return_value;
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == -1) {
            decoder_start_token_id = llama_token_bos(model);
        }

        embd_inp.clear();
        embd_inp.push_back(decoder_start_token_id);
    }

    while (n_remain != 0 && !is_antiprompt) {
        // predict
        if (!embd.empty()) {
            // Note: (n_ctx - 4) here is to match the logic for commandline prompt handling via
            // --prompt or --file which uses the same value.
            int max_embd_size = n_ctx - 4;

            // Ensure the input doesn't exceed the context size by truncating embd if necessary.
            if ((int) embd.size() > max_embd_size) {
                const int skipped_tokens = (int) embd.size() - max_embd_size;
                embd.resize(max_embd_size);
                LOG_WRN("<<input too long: skipped %d token%s>>", skipped_tokens, skipped_tokens != 1 ? "s" : "");
            }

            // try to reuse a matching prefix from the loaded session instead of re-eval (via n_past)
            if (n_session_consumed < (int)session_tokens.size()) {
                size_t i = 0;
                for ( ; i < embd.size(); i++) {
                    if (embd[i] != session_tokens[n_session_consumed]) {
                        session_tokens.resize(n_session_consumed);
                        break;
                    }

                    n_past++;
                    n_session_consumed++;

                    if (n_session_consumed >= (int)session_tokens.size()) {
                        ++i;
                        break;
                    }
                }
                if (i > 0) {
                    embd.erase(embd.begin(), embd.begin() + i);
                }
            }

            for (int i = 0; i < (int)embd.size(); i += params.n_batch) {
                int n_eval = (int)embd.size() - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                LOG_DBG("eval: %s\n", string_from(ctx, embd).c_str());

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval))) {
                    LOG_ERR("%s : failed to eval\n", __func__);
                    const llama_perf_context_data timings = llama_perf_context(ctx);
                    return_value.n_p_eval = timings.n_p_eval;
                    return_value.n_eval = timings.n_eval;
                    return_value.result = 4;
                    return return_value;
                }

                n_past += n_eval;
                LOG_DBG("n_past = %d\n", n_past);
                LOG_DBG("\n\033[31mTokens consumed so far = %d / %d \033[0m\n", n_past, n_ctx);
            }

            if (!embd.empty() && !path_session.empty()) {
                session_tokens.insert(session_tokens.end(), embd.begin(), embd.end());
                n_session_consumed = (int) session_tokens.size();
            }
        }

        embd.clear();

        if ((int)embd_inp.size() <= n_consumed) {
            // optionally save the session on first sample (for faster prompt loading next time)
            if (!path_session.empty() && need_to_save_session && !params.prompt_cache_ro) {
                need_to_save_session = false;
                llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());

                LOG_DBG("saved session to %s\n", path_session.c_str());
            } 

            if (params.prompt_cache_all == true && need_to_save_state == true && resuse_last_prompt_data == false) {
                LOG_DBG("saving last used prompt data.\n");
                need_to_save_state = false;
                if (prompt_cache_data->last_processed_prompt_state != nullptr) {
                    delete[] prompt_cache_data->last_processed_prompt_state;
                }
                const size_t state_size = llama_state_get_size(ctx);
                prompt_cache_data->last_processed_prompt_state = new uint8_t[state_size];
                prompt_cache_data->last_processed_prompt_state_size = state_size;
                llama_state_get_data(ctx, prompt_cache_data->last_processed_prompt_state, state_size);
                prompt_cache_data->last_prompt = params.prompt;
                LOG_DBG("Adding to the processed_prompt_tokens vector %d tokens from embd_inp.\n", (int)embd_inp.size());
                prompt_cache_data->processed_prompt_tokens.insert(prompt_cache_data->processed_prompt_tokens.end(), embd_inp.begin(),embd_inp.end());
            }

            const llama_token id = common_sampler_sample(smpl, ctx, -1);

            common_sampler_accept(smpl, id, true);

            // LOG_DBG("last: %s\n", string_from(ctx, smpl->prev.to_vector()).c_str());

            embd.push_back(id);

            // decrement remaining sampling budget
            --n_remain;

            LOG_DBG("n_remain: %d\n", n_remain);

            // call the token callback with the newly predicted token
            if (token_cb != NULL) {
                auto token_str = common_token_to_piece(ctx, id, include_specials);
                if (!token_cb(token_str.c_str())) {
                    break;
                }
            }

            for (auto id : embd) {
                res += common_token_to_piece(ctx, id, include_specials);
            }
        } else {
            // some user input remains from prompt or interaction, forward it to processing
            LOG_DBG("embd_inp.size(): %d, n_consumed: %d\n", (int) embd_inp.size(), n_consumed);
            while ((int) embd_inp.size() > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);

                // push the prompt in the sampling context in order to apply repetition penalties later
                // for the prompt, we don't apply grammar rules
                common_sampler_accept(smpl, embd_inp[n_consumed], /* accept_grammar= */ false);

                ++n_consumed;
                if ((int) embd.size() >= params.n_batch) {
                    break;
                }
            }
        }

        // if not currently processing queued inputs;
        if ((int) embd_inp.size() <= n_consumed) {
            // check for reverse prompt in the last n_prev tokens
            if (!params.antiprompt.empty()) {
                const int n_prev = 32;
                const std::string last_output = common_sampler_prev_str(smpl, ctx, n_prev);

                is_antiprompt = false;
                // Check if each of the reverse prompts appears at the end of the output.
                // If we're not running interactively, the reverse prompt might be tokenized with some following characters
                // so we'll compensate for that by widening the search window a bit.
                for (std::string & antiprompt : params.antiprompt) {
                    size_t extra_padding = 2;
                    size_t search_start_pos = last_output.length() > static_cast<size_t>(antiprompt.length() + extra_padding)
                        ? last_output.length() - static_cast<size_t>(antiprompt.length() + extra_padding)
                        : 0;

                    if (last_output.find(antiprompt, search_start_pos) != std::string::npos) {
                        is_antiprompt = true;
                        break;
                    }
                }

                // check for reverse prompt using special tokens
                llama_token last_token = common_sampler_last(smpl);
                for (std::vector<llama_token> ids : antiprompt_ids) {
                    if (ids.size() == 1 && last_token == ids[0]) {
                        is_antiprompt = true;
                        break;
                    }
                }

                if (is_antiprompt) {
                    LOG_DBG("found antiprompt: %s\n", last_output.c_str());
                }
            }
        }

        // end of generation
        if (llama_token_is_eog(model, common_sampler_last(smpl))) {
            LOG_DBG(" [end of text]\n");
            break;
        }
    }

    if (!path_session.empty() && !params.prompt_cache_ro) {
        LOG_INF("\n%s: saving final output to session file '%s'\n", __func__, path_session.c_str());
        llama_state_save_file(ctx, path_session.c_str(), session_tokens.data(), session_tokens.size());
    }

    // build up the result structure with the success code and all the timing data
    const llama_perf_context_data timings = llama_perf_context(ctx);
    const double t_end_ms = 1e-3 * ggml_time_us();
    return_value.result = 0;
    return_value.t_start_ms = timings.t_start_ms;
    return_value.t_end_ms = t_end_ms;
    return_value.t_load_ms = timings.t_load_ms;
    return_value.t_p_eval_ms = timings.t_p_eval_ms;
    return_value.t_eval_ms = timings.t_eval_ms;
    return_value.n_p_eval = timings.n_p_eval;
    return_value.n_eval = timings.n_eval;

    // copy at most out_result_size characters.
    strncpy(out_result, res.c_str(), (size_t) out_result_size - 1);
    out_result[out_result_size-1] = 0;

    common_sampler_free(smpl);
    ggml_threadpool_free(threadpool);
    ggml_threadpool_free(threadpool_batch);   

    return return_value;
}

void 
wooly_free_prompt_cache(void *prompt_cache_ptr)
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
    output.seed = prototype.sparams.seed;
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
    output.ignore_eos = prototype.sparams.ignore_eos;
    output.flash_attn = prototype.flash_attn;

    output.embedding = prototype.embedding;
    output.embd_normalize = prototype.embd_normalize;

    output.top_k = prototype.sparams.top_k;
    output.top_p = prototype.sparams.top_p;
    output.min_p = prototype.sparams.min_p;
    output.tfs_z = prototype.sparams.tfs_z;
    output.typical_p = prototype.sparams.typ_p;
    output.temp = prototype.sparams.temp;
    output.dynatemp_range = prototype.sparams.dynatemp_range;
    output.dynatemp_exponent = prototype.sparams.dynatemp_exponent;
    output.penalty_last_n = prototype.sparams.penalty_last_n;
    output.penalty_repeat = prototype.sparams.penalty_repeat;
    output.penalty_freq = prototype.sparams.penalty_freq;
    output.penalty_present = prototype.sparams.penalty_present;
    output.mirostat = prototype.sparams.mirostat;
    output.mirostat_tau = prototype.sparams.mirostat_tau;
    output.mirostat_eta = prototype.sparams.mirostat_eta;
    output.penalize_nl = prototype.sparams.penalize_nl;
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

    output->sparams.seed = simple->seed;
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
    output->sparams.ignore_eos = simple->ignore_eos;
    output->flash_attn = simple->flash_attn;

    output->embedding = simple->embedding;
    output->embd_normalize = simple->embd_normalize;

    output->sparams.top_k = simple->top_k;
    output->sparams.top_p = simple->top_p;
    output->sparams.min_p = simple->min_p;
    output->sparams.tfs_z = simple->tfs_z;
    output->sparams.typ_p = simple->typical_p;
    output->sparams.temp = simple->temp;
    output->sparams.dynatemp_range = simple->dynatemp_range;
    output->sparams.dynatemp_exponent = simple->dynatemp_exponent;
    output->sparams.penalty_last_n = simple->penalty_last_n;
    output->sparams.penalty_repeat = simple->penalty_repeat;
    output->sparams.penalty_freq = simple->penalty_freq;
    output->sparams.penalty_present = simple->penalty_present;
    output->sparams.mirostat = simple->mirostat;
    output->sparams.mirostat_tau = simple->mirostat_tau;
    output->sparams.mirostat_eta = simple->mirostat_eta;
    output->sparams.penalize_nl = simple->penalize_nl;
    if (simple->grammar != nullptr) {
        output->sparams.grammar = simple->grammar;
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
wooly_llama_n_embd(void *llama_model_ptr) 
{
    return llama_n_embd((const llama_model *)llama_model_ptr);
}

int64_t
wooly_llama_tokenize(
    void *llama_model_ptr, 
    const char* text,
    bool add_special,
    bool parse_special,
    int32_t* out_tokens,
    int64_t out_tokens_size)
{
    auto tokens = ::common_tokenize(
        (const llama_model *)llama_model_ptr, 
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
    void *llama_context_ptr, 
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
        static_cast<llama_context*>(llama_context_ptr), 
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
    void *llama_model_ptr,
    void *llama_context_ptr,
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
            batch_decode_embeddings(static_cast<llama_context *>(llama_context_ptr), batch, out, s, n_embd, embd_normalize);
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
    batch_decode_embeddings(static_cast<llama_context *>(llama_context_ptr), batch, out, s, n_embd, embd_normalize);

    return 0;
}
