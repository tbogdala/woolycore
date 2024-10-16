#ifdef __cplusplus
#include <vector>
#include <string>

extern "C"
{
#endif

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#include <stdint.h>
#include <stdbool.h>

typedef struct wooly_load_model_result {
    void* model;
    void* ctx;
    uint32_t context_length;
} wooly_load_model_result;

typedef struct wooly_predict_result {
    // 0 == success; 1 >= failure
    int32_t result;

    // a pointer to llama_predict_prompt_cache, which is opaque to the bindings.
    void* prompt_cache;
    
    // timing data
    double t_start_ms;
    double t_end_ms;
    double t_load_ms;
    double t_p_eval_ms;
    double t_eval_ms;

    int32_t n_p_eval;
    int n_eval;
} wooly_predict_result;


// A stripped down version of `llama_model_params` for the supported features
// of the wrapper library as well as for ease of dynamic binding - this library
// can *assure* the same ordering as expected.
typedef struct wooly_llama_model_params {
    // number of layers to store in VRAM
    int32_t n_gpu_layers; 
    // how to split the model across multiple GPUs
    /*enum llama_split_mode*/ int32_t split_mode; 

    // main_gpu interpretation depends on split_mode:
    // LLAMA_SPLIT_NONE: the GPU that is used for the entire model
    // LLAMA_SPLIT_ROW: the GPU that is used for small tensors and intermediate results
    // LLAMA_SPLIT_LAYER: ignored
    int32_t main_gpu;

    // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
    const float * tensor_split;

    // Keep the booleans together to avoid misalignment during copy-by-value.
    bool vocab_only;    // only load the vocabulary, no weights
    bool use_mmap;      // use mmap if possible
    bool use_mlock;     // force system to keep model in RAM
    bool check_tensors; // validate model tensor data
} wooly_llama_model_params;

// A stripped down version of `wooly_llama_context_params` for the supported features
// of the wrapper library as well as for ease of dynamic binding - this library
// can *assure* the same ordering as expected.
typedef struct wooly_llama_context_params {
    uint32_t n_ctx;             // text context, 0 = from model
    uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
    uint32_t n_ubatch;          // physical maximum batch size
    uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
    uint32_t n_threads;         // number of threads to use for generation
    uint32_t n_threads_batch;   // number of threads to use for batch processing

    /*enum llama_rope_scaling_type*/ int32_t rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
    /*enum llama_pooling_type*/ int32_t      pooling_type;      // whether to pool (sum) embedding results by sequence id

    // ref: https://github.com/ggerganov/llama.cpp/pull/2054
    float    rope_freq_base;   // RoPE base frequency, 0 = from model
    float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
    float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
    float    yarn_attn_factor; // YaRN magnitude scaling factor
    float    yarn_beta_fast;   // YaRN low correction dim
    float    yarn_beta_slow;   // YaRN high correction dim
    uint32_t yarn_orig_ctx;    // YaRN original context size
    float    defrag_thold;     // defragment the KV cache if holes/size > thold, < 0 disabled (default)

    // Keep the booleans together to avoid misalignment during copy-by-value.
    bool logits_all;  // the llama_decode() call computes all logits, not just the last one (DEPRECATED - set llama_batch.logits instead)
    bool embeddings;  // if true, extract embeddings (together with logits)
    bool offload_kqv; // whether to offload the KQV ops (including the KV cache) to GPU
    bool flash_attn;  // whether to use flash attention [EXPERIMENTAL]
} wooly_llama_context_params;

// A stripped down version of `gpt_params` for the supported features
// of the wrapper library as well as for ease of dynamic binding - this library
// can *assure* the same ordering as expected, and in particular for this struct,
// we limit it to C compatible types.
typedef struct wooly_gpt_params {
    const char* prompt;
    const char ** antiprompts;
    int32_t antiprompt_count;

    uint32_t seed;              // RNG seed
    int32_t n_threads;
    int32_t n_threads_batch;    // number of threads to use for batch processing (-1 = use n_threads)
    int32_t n_predict;          // new tokens to predict
    int32_t n_ctx;              // context size
    int32_t n_batch;            // logical batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_gpu_layers;       // number of layers to store in VRAM (-1 - use default)
    uint32_t split_mode;// how to split the model across GPUs
    int32_t main_gpu;           // the GPU that is used for scratch and small tensors
    float   tensor_split[128];  // how split tensors should be distributed across GPUs
    int32_t grp_attn_n;         // group-attention factor
    int32_t grp_attn_w;         // group-attention width
    float   rope_freq_base;     // RoPE base frequency
    float   rope_freq_scale;    // RoPE frequency scaling factor
    float   yarn_ext_factor;    // YaRN extrapolation mix factor
    float   yarn_attn_factor;   // YaRN magnitude scaling factor
    float   yarn_beta_fast;     // YaRN low correction dim
    float   yarn_beta_slow;     // YaRN high correction dim
    int32_t yarn_orig_ctx;      // YaRN original context length
    int32_t rope_scaling_type;

    bool prompt_cache_all;      // save user input and generations to prompt cache
    bool ignore_eos;            // ignore generated EOS tokens
    bool flash_attn;            // flash attention

    // embedding
    bool embedding;             // get only sentence embedding
    int32_t embd_normalize;     // normalisation for embendings (-1=none, 0=max absolute int16, 1=taxicab, 2=euclidean, >2=p-norm)
    
    /* incorporate llama_sampling_params members too*/

    int32_t     top_k;                  // <= 0 to use vocab size
    float       top_p;                  // 1.0 = disabled
    float       min_p;                  // 0.0 = disabled
    float       tfs_z;                  // 1.0 = disabled
    float       typical_p;              // 1.0 = disabled
    float       temp;                   // <= 0.0 to sample greedily, 0.0 to not output probabilities
    float       dynatemp_range;         // 0.0 = disabled
    float       dynatemp_exponent;      // controls how entropy maps to temperature in dynamic temperature sampler
    int32_t     penalty_last_n;         // last n tokens to penalize (0 = disable penalty, -1 = context size)
    float       penalty_repeat;         // 1.0 = disabled
    float       penalty_freq;           // 0.0 = disabled
    float       penalty_present;        // 0.0 = disabled
    int32_t     mirostat;               // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    float       mirostat_tau;           // target entropy
    float       mirostat_eta;           // learning rate
    bool        penalize_nl;            // consider newlines as a repeatable token

    const char* grammar;

} wooly_gpt_params;

// the token update callback for wooly_redict should return a bool indicating if prediction should continue (true),
// or if the prediction should stop (false).
typedef bool (*wooly_token_update_callback)(const char *token_str);

// loads a GGUF compatible model from the provided `fname` filepath, using the
// parameters provided to control features. if `silent_llama` is set to true,
// then attempts will be made to curb all output from the upstream llama.cpp library.
// if `silent_llama` is false, then 'normal' log level output is generated unless
// WOOLY_DEBUG is defined - which will cause maximum verbosity log output.
LLAMA_API wooly_load_model_result 
wooly_load_model(
    const char *fname, 
    wooly_llama_model_params model_params, 
    wooly_llama_context_params context_params,
    bool silent_llama);
        
// frees the memory needed by the model and unloads it from memory.
LLAMA_API void 
wooly_free_model(
    void *llama_context_ptr, 
    void *llama_model_ptr);

// gets a new set of text generation parameters that is a reduced set of
// parameters that upstream llama.cpp uses. 
LLAMA_API wooly_gpt_params 
wooly_new_gpt_params();

// run a text prediction base on the `simple_params` passed in, which is a reduced
// set of parameters upstream llama.cpp uses. `out_result` is a `char` buffer that
// should be large enough to hold the generated output, and the maximum size for
// the buffer should be passed as `out_result_size`. `prompt_cache_ptr` is a 
// pointer to the last used prompt cache from a previous `wooly_predict_result` and
// can be NULL if no cache is to be used; using the cache saves the function from
// having to process the same prompt again.
LLAMA_API wooly_predict_result 
wooly_predict(
    wooly_gpt_params simple_params, 
    void *llama_context_ptr, 
    void *llama_model_ptr, 
    bool include_specials, 
    char *out_result, 
    int64_t out_result_size,
    void* prompt_cache_ptr, 
    wooly_token_update_callback token_cb);    

// free the pointer returned in wooly_predict_result from llama_predict().
// only needed if you're not intending to use the prompt cache feature
LLAMA_API void 
wooly_free_prompt_cache(
    void *prompt_cache_ptr);

// creates a new set of model parameters based on the defaults 
// of upstream llama.cpp.
LLAMA_API wooly_llama_model_params
wooly_get_default_llama_model_params();

// creates a new set of context parameters based on the defaults 
// of upstream llama.cpp.
LLAMA_API wooly_llama_context_params
wooly_get_default_llama_context_params();

// returns the size of the embedding vectors used by the model.
LLAMA_API int32_t 
wooly_llama_n_embd(
    void *llama_model_ptr
);

// tokenizes the `text` passed in. if `out_tokens` is not null, then it
// will copy at most `out_tokens_size` tokens to the `out_tokens` buffer
// and return total number of tokens copied. if `out_tokens` is null,
// the function simply returns the number of tokens for the `text`.
LLAMA_API int64_t
wooly_llama_tokenize(
    void *llama_model_ptr, 
    const char* text,
    bool add_special,
    bool parse_special,
    int32_t* out_tokens,
    int64_t out_tokens_size
);

// detokenizes an array of tokens passed in to a string in the `out_result`
// memory buffer. if `render_specials` is true, special tokens will get
// turned into text as well. returns the number of characters written
// to the `out_result` buffer, or a negative number which the absolute
// value of is the buffer size needed to hold the result.
LLAMA_API int64_t 
wooly_llama_detokenize(
    void *llama_context_ptr, 
    bool render_specials, 
    int32_t* tokens,
    int64_t tokens_size,
    char *out_result, 
    int64_t out_result_size);    


// calculates embeddings for the given token arrays. 
//
//      `pooling_type` should correspond to `llama_pooling_type` values. 
//          As long as it's not LLAMA_POOLING_TYPE_NONE, a single embedding
//          vector will be created for each tokenized prompt passed in
//          as `token_arrays`. If no pooling, then each token gets its
//          own embedding vector.
//      `embd_normalize` uses the same values as the same-named member 
//          of `wooly_gpt_params`. `token_array_count`
//      `token_array_count` should be the number of `int32_t*` pointers
//          passed in as `token_arrays` and is essentially the number of
//          'prompts' that have been tokenzed for embedding creation.
//      `token_arrays` is an array of pointers to tokenized prompts, the
//          length of each array is passed in under `token_array_sizes`.
//      `token_array_sizes` is an array of sizes to determine the number
//          of `int32_t` tokens in each prompt passed in as `token_arrays`.
//      `output_embeddings` is a pointer to a `float` array to hold the
//          embeddings generated. The required size of this buffer changes
//          based on the `pooling_type` - if `pooling_type` is 
//          LLAMA_POOLING_TYPE_NONE, then required memory is 
//          `sizeof(float) * n_embd * total_token_count`. If there is
//          pooling, then each prompt is reduced to one embedding vector
//          so the memory required is `sizeof(float) * n_embd * total_prompts`.
//          In these expressions n_embd is the size of the embedding vector
//          used by the model and can be retrieved by calling `wooly_llama_n_embd()`.
//      `output_embeddings_size` is a safetey measure to make sure the output
//          buffer has enough space for the result.
//  The function returns 0 on success and `-(required_size)` if the output buffer
//  is not big enough; getting the abs() of the return value will yield the size needed.
LLAMA_API long
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
    int64_t output_embeddings_size
);

#ifdef __cplusplus
}

#endif