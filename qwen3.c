/* Qwen3 Transformer inference library implementation */

#include "qwen3.h"

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Memory management

void malloc_run_state(RunState* s, Config *p) {
    int all_heads_dim = p->n_heads * p->head_dim;
    int kv_dim = p->n_kv_heads * p->head_dim;

    s->x = calloc(p->dim, sizeof(float));
    s->xb = calloc(all_heads_dim, sizeof(float));
    s->hb = calloc(p->hidden_dim, sizeof(float));
    s->hb2 = calloc(p->hidden_dim, sizeof(float));
    s->xq = (QuantizedTensor) { .q = calloc(all_heads_dim, sizeof(int8_t)), .s = calloc(all_heads_dim / GS, sizeof(float)) };
    s->hq = (QuantizedTensor) { .q = calloc(p->hidden_dim, sizeof(int8_t)), .s = calloc(p->hidden_dim / GS, sizeof(float)) };
    s->q = calloc(all_heads_dim, sizeof(float));
    s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = calloc(p->vocab_size, sizeof(float));
    s->key_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));
    s->value_cache = calloc(p->n_layers * (uint64_t)p->seq_len * kv_dim, sizeof(float));

    if (!s->x || !s->xb || !s->hb || !s->hb2 || !s->q || !s->att || !s->logits || !s->key_cache || !s->value_cache) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->hb2);
    free(s->xq.q);
    free(s->xq.s);
    free(s->hq.q);
    free(s->hq.s);
    free(s->q);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}

// ----------------------------------------------------------------------------
// Quantization functions

void dequantize(QuantizedTensor *qx, float *x, int n) {
    for (int i = 0; i < n; i++)
        x[i] = qx->q[i] * qx->s[i / GS];
}

void quantize(QuantizedTensor *qx, float *x, int n) {
    for (int group = 0; group < n / GS; group++) {
        float wmax = 0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax)
                wmax = val;
        }

        float scale = wmax / 127.0f;
        qx->s[group] = scale;

        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale;
            int8_t quantized = (int8_t) round(quant_value);
            qx->q[group * GS + i] = quantized;
        }
    }
}

QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
    QuantizedTensor *res = malloc(n * sizeof(QuantizedTensor));

    for (int i = 0; i < n; i++) {
        res[i].q = (int8_t*)*ptr;
        *ptr = (int8_t*)*ptr + size_each;
        res[i].s = (float*)*ptr;
        *ptr = (float*)*ptr + size_each / GS;
    }
    return res;
}

void memory_map_weights(TransformerWeights *w, Config *p, void *ptr) {
    float *fptr = (float*) ptr;

    w->rms_att_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_ffn_weight = fptr;
    fptr += p->n_layers * p->dim;
    w->rms_final_weight = fptr;
    fptr += p->dim;
    w->q_norm_weights = fptr;
    fptr += p->n_layers * p->head_dim;
    w->k_norm_weights = fptr;
    fptr += p->n_layers * p->head_dim;

    ptr = (void *)fptr;
    w->q_tokens = init_quantized_tensors(&ptr, 1, p->vocab_size * p->dim);
    w->token_embedding_table = malloc(p->vocab_size * p->dim * sizeof(float));
    dequantize(w->q_tokens, w->token_embedding_table, p->vocab_size * p->dim);

    w->wq = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_heads * p->head_dim));
    w->wk = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wv = init_quantized_tensors(&ptr, p->n_layers, p->dim * (p->n_kv_heads * p->head_dim));
    w->wo = init_quantized_tensors(&ptr, p->n_layers, (p->n_heads * p->head_dim) * p->dim);

    w->w1 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);
    w->w2 = init_quantized_tensors(&ptr, p->n_layers, p->hidden_dim * p->dim);
    w->w3 = init_quantized_tensors(&ptr, p->n_layers, p->dim * p->hidden_dim);

    w->wcls = p->shared_classifier ? w->q_tokens : init_quantized_tensors(&ptr, 1, p->dim * p->vocab_size);
}

void read_checkpoint(char *checkpoint, Config *config, TransformerWeights* weights, float** data, ssize_t* file_size, int ctx_length) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open checkpoint %s\n", checkpoint); exit(EXIT_FAILURE); }

    #if defined _WIN32
        _fseeki64(file, 0, SEEK_END);
        *file_size = _ftelli64(file);
    #else
        fseek(file, 0, SEEK_END);
        *file_size = ftell(file);
    #endif

    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, fileno(file), 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    fclose(file);

    memcpy(config, *data, sizeof(Config));
    if (config->magic_number != 0x616a6331) { fprintf(stderr, "File %s is not a qwen3.c checkpoint\n", checkpoint); exit(EXIT_FAILURE); }
    if (config->version != 1) { fprintf(stderr, "Checkpoint %s is version %d, need version 1\n", checkpoint, config->version); exit(EXIT_FAILURE); }

    if (ctx_length != 0 && ctx_length <= config->seq_len)
        config->seq_len = ctx_length;

    GS = config->group_size;

    void *weights_ptr = ((char *)*data) + 256;
    memory_map_weights(weights, config, weights_ptr);
}

void build_transformer(Transformer *t, char *checkpoint_path, int ctx_length) {
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->data, &t->file_size, ctx_length);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer *t) {
    free(t->weights.q_tokens);
    free(t->weights.token_embedding_table);
    free(t->weights.wq);
    free(t->weights.wk);
    free(t->weights.wv);
    free(t->weights.wo);
    free(t->weights.w1);
    free(t->weights.w2);
    free(t->weights.w3);
    if(t->weights.wcls != t->weights.q_tokens) free(t->weights.wcls);
    if (t->data != MAP_FAILED) munmap(t->data, t->file_size);
    free_run_state(&t->state);
}

// ----------------------------------------------------------------------------
// Neural net blocks

void rmsnorm(float *o, float *x, float *weight, int size) {
    float ss = 0;
    for (int j = 0; j < size; j++)
        ss += x[j] * x[j];

    ss = 1.0f / sqrtf((ss / size) + 1e-6f);

    for (int j = 0; j < size; j++)
        o[j] = weight[j] * (ss * x[j]);
}

void softmax(float *x, int size) {
    float max_val = 0;
    for (int i = 0; i < size; i++)
        if (x[i] > max_val)
            max_val = x[i];

    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++)
        x[i] /= sum;
}

void matmul(float *xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    int i;
    #pragma omp parallel for
    for (i = 0; i < d; i++) {
        float val = 0;
        int in = i * n;

        for (int j = 0; j <= n - GS; j += GS) {
            int32_t ival = 0;
            for (int k = 0; k < GS; k++)
                ival += x->q[j + k] * w->q[in + j + k];

            val += ((float) ival) * w->s[(in + j) / GS] * x->s[j / GS];
        }

        xout[i] = val;
    }
}

float *forward(Transformer *transformer, int token, int pos) {
    Config *p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int kv_mul = p->n_heads / p->n_kv_heads;
    int all_heads_dim = p->n_heads * p->head_dim;

    memcpy(s->x, w->token_embedding_table + token * p->dim, p->dim * sizeof(float));

    for (int l = 0; l < p->n_layers; l++) {
        uint64_t loff = l * (uint64_t)p->seq_len * kv_dim;

        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        rmsnorm(s->xb, s->x, w->rms_att_weight + l * p->dim, p->dim);

        quantize(&s->xq, s->xb, p->dim);
        matmul(s->q, &s->xq, w->wq + l, p->dim, all_heads_dim);
        matmul(s->k, &s->xq, w->wk + l, p->dim, kv_dim);
        matmul(s->v, &s->xq, w->wv + l, p->dim, kv_dim);

        for (int h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;

            rmsnorm(q, q, w->q_norm_weights + l * p->head_dim, p->head_dim);
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = q[j];
                float y = q[j + p->head_dim/2];

                q[j] = x * cos_freq - y * sin_freq;
                q[j + p->head_dim/2] = x * sin_freq + y * cos_freq;
            }
        }

        for (int h = 0; h < p->n_kv_heads; h++) {
            float *k = s->k + h * p->head_dim;

            rmsnorm(k, k, w->k_norm_weights + l * p->head_dim, p->head_dim);
            for (int j = 0; j < p->head_dim/2; j++) {
                float freq = powf(1e6, -(float)j / (p->head_dim/2));
                float cos_freq = cosf(pos * freq), sin_freq = sinf(pos * freq);

                float x = k[j];
                float y = k[j + p->head_dim/2];

                k[j] = x * cos_freq - y * sin_freq;
                k[j + p->head_dim/2] = x * sin_freq + y * cos_freq;
            }
        }

        int h;
        #pragma omp parallel for
        for (h = 0; h < p->n_heads; h++) {
            float *q = s->q + h * p->head_dim;
            float *att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float *k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                float score = 0;
                for (int i = 0; i < p->head_dim; i++)
                    score += q[i] * k[i];

                att[t] = score / sqrtf(p->head_dim);
            }

            softmax(att, pos + 1);

            float *xb = s->xb + h * p->head_dim;
            memset(xb, 0, p->head_dim * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float *v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * p->head_dim;
                for (int i = 0; i < p->head_dim; i++)
                    xb[i] += att[t] * v[i];
            }
        }

        quantize(&s->xq, s->xb, all_heads_dim);
        matmul(s->xb, &s->xq, w->wo + l, all_heads_dim, p->dim);

        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];

        rmsnorm(s->xb, s->x, w->rms_ffn_weight + l * p->dim, p->dim);

        quantize(&s->xq, s->xb, p->dim);
        matmul(s->hb, &s->xq, w->w1 + l, p->dim, p->hidden_dim);
        matmul(s->hb2, &s->xq, w->w3 + l, p->dim, p->hidden_dim);

        for (int i = 0; i < p->hidden_dim; i++)
            s->hb[i] *= s->hb2[i] * (1.0f / (1.0f + expf(-s->hb[i])));

        quantize(&s->hq, s->hb, p->hidden_dim);
        matmul(s->xb, &s->hq, w->w2 + l, p->hidden_dim, p->dim);

        for (int i = 0; i < p->dim; i++)
            s->x[i] += s->xb[i];
    }

    rmsnorm(s->x, s->x, w->rms_final_weight, p->dim);

    quantize(&s->xq, s->x, p->dim);
    matmul(s->logits, &s->xq, w->wcls, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer

void load_prompt_template(char *checkpoint_path, char *out_template, int with_system_prompt, int enable_thinking) {
    char prompt_path[1024];

    strcpy(prompt_path, checkpoint_path);
    if (with_system_prompt)
        strcat(prompt_path, enable_thinking ? ".template.with-system-and-thinking" : ".template.with-system");
    else
        strcat(prompt_path, enable_thinking ? ".template.with-thinking" : ".template");

    memset(out_template, 0, 1024);
    FILE *file = fopen(prompt_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load prompt template %s\n", prompt_path); exit(EXIT_FAILURE); }
    fread(out_template, 1024, 1, file);
    fclose(file);
}

void build_tokenizer(Tokenizer *t, char *checkpoint_path, int vocab_size, int enable_thinking) {
    char tokenizer_path[1024];

    strcpy(tokenizer_path, checkpoint_path);
    strcat(tokenizer_path, ".tokenizer");

    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->merge_scores = (float *)malloc(vocab_size * sizeof(float));

    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "Couldn't load tokenizer model %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    fread(&t->bos_token_id, sizeof(int), 1, file);
    fread(&t->eos_token_id, sizeof(int), 1, file);

    int len;

    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->merge_scores + i, sizeof(float), 1, file) != 1) {
            t->vocab[i] = (char *)malloc(1);
            t->vocab[i][0] = 0;
        } else {
            fread(&len, sizeof(int), 1, file);
            t->vocab[i] = (char *)malloc(len + 1);
            fread(t->vocab[i], 1, len, file);
            t->vocab[i][len] = 0;
        }
    }
    fclose(file);

    load_prompt_template(checkpoint_path, t->prompt_template, 0, enable_thinking);
    load_prompt_template(checkpoint_path, t->system_prompt_template, 1, enable_thinking);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->merge_scores);
}

char *decode(Tokenizer *t, int token) {
    return t->vocab[token];
}

int str_lookup(char *str, char **vocab, int vocab_size) {
    for (int i = 0; i < vocab_size; i++)
        if (!strcmp(str, vocab[i]))
            return i;

    return -1;
}

void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    char *str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    char special_token[64 + 1];

    *n_tokens = 0;

    for (char *c = text; *c != 0; c++) {
        int id, found_special_token = 0;

        str_buffer[0] = *c;
        str_buffer[1] = 0;

        if (*c == '<') {
          int end_of_token_pos = -1;
          found_special_token = 0;
          for (int k = 0; *c != 0 && k < 64; k++) {
              if (c[k] == '>') {
                  end_of_token_pos = k;
                  break;
              }
          }

          if (end_of_token_pos != -1) {
              strncpy(special_token, c, end_of_token_pos + 1);
              special_token[end_of_token_pos + 1] = 0;

              id = str_lookup(special_token, t->vocab, t->vocab_size);
              if (id != -1) {
                  c += end_of_token_pos;
                  found_special_token = 1;
              }
          }
        }

        if (!found_special_token)
            id = str_lookup(str_buffer, t->vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            (*n_tokens)++;
        }
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->vocab, t->vocab_size);

            if (id != -1 && t->merge_scores[id] > best_score) {
                best_score = t->merge_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
            break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
            tokens[i] = tokens[i + 1];

        (*n_tokens)--;
    }

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler

int sample_argmax(float *probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float *probabilities, int n, float coin) {
    float cdf = 0;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf)
            return i;
    }
    return n - 1;
}

int compare(const void *a, const void *b) {
    ProbIndex *a_ = (ProbIndex *) a;
    ProbIndex *b_ = (ProbIndex *) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float *probabilities, int n, float topp, ProbIndex *probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf)
            return probindex[i].index;
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler *sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler *sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler *sampler, float *logits) {
    if (sampler->temperature == 0) {
        return sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            return sample_mult(logits, sampler->vocab_size, coin);
        } else {
            return sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
}

// ----------------------------------------------------------------------------
// Chat API for GUI

int chat_init(ChatState *state, const char *checkpoint_path, const char *system_prompt,
              float temperature, float topp, unsigned long long rng_seed, 
              int enable_thinking, int ctx_length) {
    
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0) temperature = 0;
    if (topp < 0 || 1.0 < topp) topp = 0.9;

    // Build transformer
    build_transformer(&state->transformer, (char*)checkpoint_path, ctx_length);
    
    // Build tokenizer
    build_tokenizer(&state->tokenizer, (char*)checkpoint_path, state->transformer.config.vocab_size, enable_thinking);
    
    // Build sampler
    build_sampler(&state->sampler, state->transformer.config.vocab_size, temperature, topp, rng_seed);
    
    // Initialize chat state
    state->pos = 0;
    state->prompt_idx = 0;
    state->user_turn = 1;
    state->num_prompt_tokens = 0;
    state->prompt_tokens = (int *)malloc(PROMPT_BUFFER_SIZE * sizeof(int));
    state->current_token = 0;
    state->next_token = 0;
    state->output_buffer[0] = 0;
    state->output_len = 0;
    state->generation_done = 1;
    
    // Copy system prompt if provided
    if (system_prompt) {
        state->system_prompt = (char*)malloc(strlen(system_prompt) + 1);
        strcpy(state->system_prompt, system_prompt);
    } else {
        state->system_prompt = NULL;
    }
    
    return 0;
}

void chat_free(ChatState *state) {
    free_sampler(&state->sampler);
    free_tokenizer(&state->tokenizer);
    free_transformer(&state->transformer);
    free(state->prompt_tokens);
    if (state->system_prompt) free(state->system_prompt);
}

void chat_submit_prompt(ChatState *state, const char *user_prompt) {
    char rendered_prompt[PROMPT_BUFFER_SIZE];

    // Render prompt with template
    if (state->pos == 0 && state->system_prompt) {
        sprintf(rendered_prompt, state->tokenizer.system_prompt_template, state->system_prompt, user_prompt);
    } else {
        sprintf(rendered_prompt, state->tokenizer.prompt_template, user_prompt);
    }

    // Encode the rendered prompt
    encode(&state->tokenizer, rendered_prompt, state->prompt_tokens, &state->num_prompt_tokens);

    // Reset generation state (but keep pos to maintain history!)
    state->prompt_idx = 0;  // reset prompt processing index, not pos
    state->user_turn = 0;
    state->output_buffer[0] = 0;
    state->output_len = 0;
    state->generation_done = 0;
}

const char* chat_generate_next(ChatState *state) {
    if (state->generation_done) return NULL;

    // Check context window
    if (state->pos >= state->transformer.config.seq_len) {
        state->generation_done = 1;
        return NULL;
    }

    int token;

    // Determine token to process
    if (state->prompt_idx < state->num_prompt_tokens) {
        // Still processing the input prompt, force the next prompt token
        token = state->prompt_tokens[state->prompt_idx];
        state->prompt_idx++;
    } else {
        // Otherwise use the next token sampled from previous turn
        token = state->next_token;
    }

    // Forward pass
    float *logits = forward(&state->transformer, token, state->pos);
    state->next_token = sample(&state->sampler, logits);
    state->pos++;

    // Check for end of generation
    if (state->prompt_idx >= state->num_prompt_tokens) {
        if (token == state->tokenizer.bos_token_id || token == state->tokenizer.eos_token_id) {
            state->generation_done = 1;
            state->user_turn = 1;
            return NULL;
        }

        if (state->next_token != state->tokenizer.bos_token_id &&
            state->next_token != state->tokenizer.eos_token_id) {
            return decode(&state->tokenizer, state->next_token);
        }
    }

    return "";  // Still processing prompt
}

int chat_is_done(ChatState *state) {
    return state->generation_done;
}

void chat_reset(ChatState *state) {
    state->pos = 0;
    state->prompt_idx = 0;
    state->user_turn = 1;
    state->num_prompt_tokens = 0;
    state->output_buffer[0] = 0;
    state->output_len = 0;
    state->generation_done = 1;

    // Clear KV cache
    Config *p = &state->transformer.config;
    int kv_dim = p->n_kv_heads * p->head_dim;
    memset(state->transformer.state.key_cache, 0, p->n_layers * (uint64_t)p->seq_len * kv_dim * sizeof(float));
    memset(state->transformer.state.value_cache, 0, p->n_layers * (uint64_t)p->seq_len * kv_dim * sizeof(float));
}
