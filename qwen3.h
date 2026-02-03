/* Qwen3 Transformer inference library header */
#ifndef _QWEN3_H_
#define _QWEN3_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// Maximum input prompt buffer size
#define PROMPT_BUFFER_SIZE 32768

// ----------------------------------------------------------------------------
// Transformer model structures

typedef struct {
    int magic_number;
    int version;
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
    int head_dim;
    int shared_classifier;
    int group_size;
} Config;

typedef struct {
    int8_t *q;
    float *s;
} QuantizedTensor;

typedef struct {
    QuantizedTensor *q_tokens;
    float *token_embedding_table;
    float *rms_att_weight;
    float *rms_ffn_weight;
    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;
    float *q_norm_weights;
    float *k_norm_weights;
    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;
    float *rms_final_weight;
    QuantizedTensor *wcls;
} TransformerWeights;

typedef struct {
    float *x;
    float *xb;
    float *hb;
    float *hb2;
    QuantizedTensor xq;
    QuantizedTensor hq;
    float *q;
    float *k;
    float *v;
    float *att;
    float *logits;
    float *key_cache;
    float *value_cache;
} RunState;

typedef struct {
    Config config;
    TransformerWeights weights;
    RunState state;
    float *data;
    ssize_t file_size;
} Transformer;

typedef struct {
    char **vocab;
    float *merge_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned int bos_token_id;
    unsigned int eos_token_id;
    char prompt_template[1024];
    char system_prompt_template[1024];
} Tokenizer;

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex *probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

// ----------------------------------------------------------------------------
// Chat state for GUI interaction

typedef struct {
    Transformer transformer;
    Tokenizer tokenizer;
    Sampler sampler;

    // Chat state
    int pos;                    // current position in sequence (absolute position for KV cache)
    int prompt_idx;             // index for processing current prompt tokens
    int user_turn;              // 1 if it's user's turn
    int num_prompt_tokens;
    int *prompt_tokens;
    int current_token;
    int next_token;
    char *system_prompt;

    // Output buffer for streaming
    char output_buffer[PROMPT_BUFFER_SIZE];
    int output_len;
    int generation_done;
} ChatState;

// ----------------------------------------------------------------------------
// API Functions

// Initialize the chat state with model
int chat_init(ChatState *state, const char *checkpoint_path, const char *system_prompt,
              float temperature, float topp, unsigned long long rng_seed, 
              int enable_thinking, int ctx_length);

// Free the chat state
void chat_free(ChatState *state);

// Submit user prompt and start generation
void chat_submit_prompt(ChatState *state, const char *user_prompt);

// Generate next token (non-blocking, returns token string or NULL if done)
const char* chat_generate_next(ChatState *state);

// Check if generation is complete
int chat_is_done(ChatState *state);

// Reset chat context
void chat_reset(ChatState *state);

// Internal functions exposed for library use
void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length);
void free_transformer(Transformer *t);
void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking);
void free_tokenizer(Tokenizer *t);
void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed);
void free_sampler(Sampler *sampler);
void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens);
char *decode(Tokenizer *t, int token);
float *forward(Transformer *transformer, int token, int pos);
int sample(Sampler *sampler, float *logits);

#endif /* _QWEN3_H_ */
