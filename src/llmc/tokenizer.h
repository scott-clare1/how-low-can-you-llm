/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#ifndef TOKENIZER
#define TOKENIZER

#include <_types/_uint32_t.h>
#include <stdint.h>
#include <ctype.h>
#include <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
} Tokenizer;


typedef struct {
    int num_input_tokens;
    uint32_t* input_tokens;
} EncodedInput;

void safe_printf(const char *piece);

void tokenizer_init(Tokenizer *tokenizer, const char *filename);

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id);

void tokenizer_free(Tokenizer *tokenizer);

EncodedInput tokenizer_encode(Tokenizer *tokenizer, char* input);

#endif
