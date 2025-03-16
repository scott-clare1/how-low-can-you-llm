#include "tokenizer.h"
#include <sys/types.h>
#include <stddef.h>
#include <stdio.h>
#include "inference.h"
#include "utils.c"
#include <math.h>

#include <stddef.h>
#include <stdint.h>


void encoder_forward(float* out, int* inp, float* wte, float* wpe, int T, int C, int token_count) {
    // out is (T,C). At each position (t), a C-dimensional vector summarizing token & position
    // inp is (T) of integers, holding the token ids at each (t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    // seek to the output position in out[t,:]
    float* out_bt = out + 0 * T * C + token_count * C;
    // get the index of the token at inp[t]
    int ix = inp[0 * T + token_count];
    // seek to the position in wte corresponding to the token
    float* wte_ix = wte + ix * C;
    // seek to the position in wpe corresponding to the position
    float* wpe_t = wpe + token_count * C;
    // add the two vectors and store the result in out[t,:]
    for (int i = 0; i < C; i++) {
        out_bt[i] = wte_ix[i] + wpe_t[i];
    }
}


void layernorm_forward(float* out, float* inp, float* weight, float* bias,
                       int C, int token_count) {
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (T,C) of the activations
    // mean and rstd are (T) buffers, to be used later in backward pass
    // at each position (t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;

    // seek to the input position inp[t,:]
    float* x = inp + token_count * C;
    // calculate the mean
    float m = 0.0f;
    for (int i = 0; i < C; i++) {
        m += x[i];
    }

    m = m/C;
    // calculate the variance (without any bias correction)
    float v = 0.0f;
    for (int i = 0; i < C; i++) {
        float xshift = x[i] - m;
        v += xshift * xshift;
    }
    v = v/C;
    // calculate the rstd (reciprocal standard deviation)
    float s = 1.0f / sqrtf(v + eps);
    // seek to the output position in out[t,:]
    float* out_bt = out + token_count * C;
    for (int i = 0; i < C; i++) {
        float n = (s * (x[i] - m)); // normalize
        float o = n * weight[i] + bias[i]; // scale and shift
        out_bt[i] = o; // write
    }
}


void matmul_forward(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int C, int OC, int token_count) {
    // the most naive implementation of matrix multiplication
    // this serves as an algorithmic reference, and as a fallback for
    // unfriendly input shapes inside matmul_forward(), below.
    for (int o = 0; o < OC; o++) {
        float val = (bias != NULL) ? bias[o] : 0.0f;
        for (int i = 0; i < C; i++) {
            val += inp[token_count * C + i] * weight[o*C + i];
        }
        out[token_count * OC + o] = val;
    }
}


void attention_forward(float* out, float* preatt, float* att,
                       float* inp,
                       int T, int C, int NH, int token_count) {
    // input is (T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int h = 0; h < NH; h++) {
        float* query_t = inp  + token_count * C3 + h * hs;
        float* preatt_bth = preatt + h*T*T + token_count*T;
        float* att_bth = att + h*T*T + token_count*T;

        // pass 1: calculate query dot key and maxval
        float maxval = -10000.0f; // TODO something better
        for (int t2 = 0; t2 <= token_count; t2++) {
            float* key_t2 = inp + t2 * C3 + h * hs + C; // +C because it's key

            // (query_t) dot (key_t2)
            float val = 0.0f;
            for (int i = 0; i < hs; i++) {
                val += query_t[i] * key_t2[i];
            }
            val *= scale;
            if (val > maxval) {
                maxval = val;
            }

            preatt_bth[t2] = val;
        }

        // pass 2: calculate the exp and keep track of sum
        // maxval is being calculated and subtracted only for numerical stability
        float expsum = 0.0f;
        for (int t2 = 0; t2 <= token_count; t2++) {
            float expv = expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
        }
        float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

        // pass 3: normalize to get the softmax
        for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= token_count) {
                att_bth[t2] *= expsum_inv;
            } else {
                // causal attention mask. not strictly necessary to set to zero here
                // only doing this explicitly for debugging and checking to PyTorch
                att_bth[t2] = 0.0f;
            }
        }

        // pass 4: accumulate weighted values into the output of attention
        float* out_bth = out + token_count * C + h * hs;
        for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
        for (int t2 = 0; t2 <= token_count; t2++) {
            float* value_t2 = inp + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++) {
                out_bth[i] += att_btht2 * value_t2[i];
            }
        }
    }
}


#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float* out, float* inp, int N) {
    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
        float x = inp[i];
        float cube = 0.044715f * x * x * x;
        out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
}


void residual_forward(float* out, float* inp1, float* inp2, int N) {
    for (int i = 0; i < N; i++) {
        out[i] = inp1[i] + inp2[i];
    }
}


void softmax_forward(float* probs, float* logits, int B, int V, int Vp, int token_count) {
    // output: probs are (T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257
    //
    // probs <- softmax(logits)

    float* logits_t = logits + (token_count) * Vp;
    float* probs_t = probs + (token_count) * Vp;

    // maxval is only calculated and subtracted for numerical stability
    float maxval = -10000.0f; // TODO something better
    for (int i = 0; i < V; i++) {
        if (logits_t[i] > maxval) {
            maxval = logits_t[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < V; i++) {
        probs_t[i] = expf(logits_t[i] - maxval);
        sum += probs_t[i];
    }

    for (int i = 0; i < V; i++) {
        probs_t[i] /= sum;
    }

    for (int i = V; i < Vp; i++) {
        probs_t[i] = 0.0f;
    }
}



void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config) {
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C; // wte
    param_sizes[1] = maxT * C; // wpe
    param_sizes[2] = L * C; // ln1w
    param_sizes[3] = L * C; // ln1b
    param_sizes[4] = L * (3 * C) * C; // qkvw
    param_sizes[5] = L * (3 * C); // qkvb
    param_sizes[6] = L * C * C; // attprojw
    param_sizes[7] = L * C; // attprojb
    param_sizes[8] = L * C; // ln2w
    param_sizes[9] = L * C; // ln2b
    param_sizes[10] = L * (4 * C) * C; // fcw
    param_sizes[11] = L * (4 * C); // fcb
    param_sizes[12] = L * C * (4 * C); // fcprojw
    param_sizes[13] = L * C; // fcprojb
    param_sizes[14] = C; // lnfw
    param_sizes[15] = C; // lnfb
}

// allocate memory for the parameters and point the individual tensors to the right places
float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes) {
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float* params_memory = (float*)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb
    };
    float* params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}


void fill_in_activation_sizes(size_t* act_sizes, GPT2Config config, int T) {
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = T * C; // encoded
    act_sizes[1] = L * T * C; // ln1
    act_sizes[2] = L * T; // ln1_mean
    act_sizes[3] = L * T; // ln1_rstd
    act_sizes[4] = L * T * 3 * C; // qkv
    act_sizes[5] = L * T * C; // atty
    act_sizes[6] = L * NH * T * T; // preatt
    act_sizes[7] = L * NH * T * T; // att
    act_sizes[8] = L * T * C; // attproj
    act_sizes[9] = L * T * C; // residual2
    act_sizes[10] = L * T * C; // ln2
    act_sizes[11] = L * T; // ln2_mean
    act_sizes[12] = L * T; // ln2_rstd
    act_sizes[13] = L * T * 4 * C; // fch
    act_sizes[14] = L * T * 4 * C; // fch_gelu
    act_sizes[15] = L * T * C; // fcproj
    act_sizes[16] = L * T * C; // residual3
    act_sizes[17] = T * C; // lnf
    act_sizes[18] = T; // lnf_mean
    act_sizes[19] = T; // lnf_rstd
    act_sizes[20] = T * Vp; // logits
    act_sizes[21] = T * Vp; // probs
    act_sizes[22] = T; // losses
}

float* malloc_and_point_activations(ActivationTensors* acts, size_t* act_sizes) {
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        num_activations += act_sizes[i];
    }
    float* acts_memory = (float*)mallocCheck(num_activations * sizeof(float));
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses
    };
    float* acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}


void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(1); }
    if (model_header[1] != 3) {
        printf("Bad version in model file\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes,  model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters;

    // read in all the parameters from file
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes);
    freadCheck(model->params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
}

void gpt2_forward(GPT2* model, int* inputs, size_t T, int token_count) {

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    // convenience parameters (size_t to help prevent int overflow)
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    memcpy(model->inputs, inputs, T * sizeof(int));

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, inputs, params.wte, params.wpe, T, C, token_count - 1); // encoding goes into residual[0]

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * T;
        float* l_qkv = acts.qkv + l * T * 3*C;
        float* l_atty = acts.atty + l * T * C;
        float* l_preatt = acts.preatt + l * NH * T * T;
        float* l_att = acts.att + l * NH * T * T;
        float* l_attproj = acts.attproj + l * T * C;
        float* l_residual2 = acts.residual2 + l * T * C;
        float* l_ln2 = acts.ln2 + l * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * T;
        float* l_fch = acts.fch + l * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * T * 4*C;
        float* l_fcproj = acts.fcproj + l * T * C;
        float* l_residual3 = acts.residual3 + l * T * C;

        // now do the forward pass
        layernorm_forward(l_ln1, residual, l_ln1w, l_ln1b, C, token_count-1);
        matmul_forward(l_qkv, l_ln1, l_qkvw, l_qkvb, C, 3*C, token_count-1);
        attention_forward(l_atty, l_preatt, l_att, l_qkv, T, C, NH, token_count-1);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, C, C, token_count-1);
        residual_forward(l_residual2, residual, l_attproj, T*C);
        layernorm_forward(l_ln2,  l_residual2, l_ln2w, l_ln2b, C, token_count-1);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, C, 4*C, token_count-1);
        gelu_forward(l_fch_gelu, l_fch, T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, 4*C, C, token_count-1);
        residual_forward(l_residual3, l_residual2, l_fcproj, T*C);
    }
    residual = acts.residual3 + (L-1) * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf,  residual, params.lnfw, params.lnfb, C, token_count-1);
    matmul_forward(acts.logits, acts.lnf, params.wte, NULL, C, Vp, token_count-1);
    softmax_forward(acts.probs, acts.logits, 1, V, Vp, token_count-1);
}

unsigned int random_u32(uint64_t *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

void gpt2_free(GPT2 *model) {
    free(model->params_memory);
    free(model->grads_memory);
    free(model->acts_memory);
    free(model->grads_acts_memory);
    free(model->inputs);
}


char* get_next_token(
    int* gen_tokens,
    GPT2* model,
    uint64_t* rng_state,
    int n_tokens,
    Tokenizer tokenizer,
    int token_count
) {
    gpt2_forward(model, gen_tokens, n_tokens, token_count);
    float* probs = model->acts.probs + (token_count-1) * model->config.padded_vocab_size;
    float coin = random_f32(rng_state);

    int next_token = sample_mult(probs, model->config.vocab_size, coin);

    gen_tokens[token_count] = next_token;

    if (tokenizer.init_ok) {
        return tokenizer_decode(&tokenizer, next_token);
    } else {
        printf("Failed to tokenize: %d\n.", next_token);
        exit(1);
    }
}
