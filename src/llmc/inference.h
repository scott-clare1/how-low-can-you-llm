#ifndef INFERENCE
#define INFERENCE

#include "tokenizer.h"
#include <sys/types.h>
#include <stddef.h>
#include <stdio.h>

#include <stddef.h>
#include <stdint.h>

// ----------------------------------------------------------------------------
// GPT-2 model definition


typedef struct {
    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} GPT2Config;


// the parameters of the model
#define NUM_PARAMETER_TENSORS 16
typedef struct {
    float* wte; // (V, C)
    float* wpe; // (maxT, C)
    float* ln1w; // (L, C)
    float* ln1b; // (L, C)
    float* qkvw; // (L, 3*C, C)
    float* qkvb; // (L, 3*C)
    float* attprojw; // (L, C, C)
    float* attprojb; // (L, C)
    float* ln2w; // (L, C)
    float* ln2b; // (L, C)
    float* fcw; // (L, 4*C, C)
    float* fcb; // (L, 4*C)
    float* fcprojw; // (L, C, 4*C)
    float* fcprojb; // (L, C)
    float* lnfw; // (C)
    float* lnfb; // (C)
} ParameterTensors;


#define NUM_ACTIVATION_TENSORS 23
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* qkv; // (L, B, T, 3*C)
    float* atty; // (L, B, T, C)
    float* preatt; // (L, B, NH, T, T)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* logits; // (B, T, V)
    float* probs; // (B, T, V)
    float* losses; // (B, T)
} ActivationTensors;


typedef struct {
    GPT2Config config;
    // the weights (parameters) of the model, and their sizes
    ParameterTensors params;
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    float* params_memory;
    size_t num_parameters;
    // gradients of the weights
    ParameterTensors grads;
    float* grads_memory;
    // the activations of the model, and their sizes
    ActivationTensors acts;
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    float* acts_memory;
    size_t num_activations;
    // gradients of the activations
    ActivationTensors grads_acts;
    float* grads_acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass
    int* inputs; // the input tokens for the current forward pass
} GPT2;

void encoder_forward(float* out, int* inp, float* wte, float* wpe, int T, int C, int token_count);


void layernorm_forward(float* out, float* inp, float* weight, float* bias,
                       int C, int token_count);


void matmul_forward(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int C, int OC, int token_count);


void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int T, int C, int NH, int token_count);


#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N);


void residual_forward(float* out, float* inp1, float* inp2, int N);

void softmax_forward(float* probs, float* logits, int B, int V, int Vp, int token_count);

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config);

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes);


void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int T);

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes);

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path);

void gpt2_forward(GPT2* model, int* inputs, size_t T, int token_count);

unsigned int random_u32(uint64_t *state);
float random_f32(uint64_t *state);

int sample_mult(float* probabilities, int n, float coin);

void gpt2_free(GPT2 *model);

char* get_next_token(
    int* gen_tokens,
    GPT2* model,
    uint64_t* rng_state,
    int n_tokens,
    Tokenizer tokenizer,
    int token_count
);

#endif
