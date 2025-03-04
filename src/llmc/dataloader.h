/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#include "utils.h"
#include "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t num_tokens; // total number of tokens
    size_t shard_num_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* tokens_file;
    // data buffers
    uint16_t* buffer; // we fread data from file into this buffer
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    int should_shuffle;
    int* shard_indices;
    int* intra_shard_indices;
    // sizes in bytes
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    int64_t file_size_bytes;
} DataLoader;

int64_t dataloader_load_shard_(DataLoader *loader, int shard_index);

void prepare_intra_shard_indices_(DataLoader *loader);

void dataloader_advance_(DataLoader *loader);

void dataloader_init(DataLoader *loader,
                     const char* filename_pattern,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes,
                     int should_shuffle);

void dataloader_load_batch(DataLoader* loader);

void dataloader_next_batch(DataLoader *loader);


void dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx);
void dataloader_reset(DataLoader *loader);

void dataloader_free(DataLoader *loader);

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Many evals (like) HellaSwag and MMLU are multiple-choice
// where there are 4 possible continuations and a label for the correct one
// We want to load and serve these style of evals
/*
Copy pasting the section on the eval datafile format, from data_common.py:
- First comes a header with 256 int32s
- The examples follow, each example is a stream of uint16_t:
    - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
    - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
    - <EXAMPLE_INDEX>, the index of the example in the dataset
    - <LABEL>, the index of the correct completion
    - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
    - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
    - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
*/

// for now, could relax later
#define ASSUMED_NUM_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    int num_examples; // in total across all processes
    int num_batches; // to process the entire dataset across all processes
    int start_example_index; // the assignment of work for this process, start
    int end_example_index; // and end. start is inclusive, end is exclusive
    int current_example_index; // the next example we would read
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    char* mask; // mask=1 at all completion token locations
    int* label; // the correct completion labels
    int num_completions; // number of completions for this example
} EvalLoader;

void evalloader_reset(EvalLoader *loader);

void evalloader_init(EvalLoader *loader,
                     const char* filename,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes);

void evalloader_next_example_(EvalLoader *loader, int example_batch_index);

void evalloader_next_batch(EvalLoader *loader);

int evalloader_stat_losses(EvalLoader *loader, float* losses);

void evalloader_free(EvalLoader *loader);

#endif // DATALOADER_H
