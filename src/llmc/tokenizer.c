/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

// #include <cstring>
#include <_types/_uint32_t.h>
#include <malloc/_malloc.h>
#include <stdint.h>
#include <ctype.h>
#include <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"
#include "tokenizer.h"

// ----------------------------------------------------------------------------


void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %u!\n", token_id);
        return NULL;
    }
}

int contains_next_substring(
    char* next,
    Tokenizer* tokenizer
) {
    for (int j = 0; j < tokenizer->vocab_size; j++) {
        if (strcmp(next, tokenizer->token_table[j]) == 0) {
            return 1;
        }
    }
    return 0;
}


int get_string_size(char* input) {
    char * t; // first copy the pointer to not change the original
    int size = 0;

    for (t = input; *t != '\0'; t++) {
        size++;
    }
    return size;
}


EncodedInput tokenizer_encode(Tokenizer *tokenizer, char* input) {
    int size = get_string_size(input);

    uint32_t* tokens = (uint32_t*)malloc(size * sizeof(uint32_t));

    char* substring = (char*)malloc(size * sizeof(char));
    char* next = (char*)malloc(size * sizeof(char));
    int counter = 0;
    int token_count = 0;
    next[counter] = input[0];
    for (int c = 0; c < size; c++) {
        substring[counter] = input[c];

        next[counter+1] = input[c+1];
        counter++;

        for (int i = 0; i < tokenizer->vocab_size; i++) {
            if (strcmp(substring, tokenizer->token_table[i]) == 0) {
                if (contains_next_substring(next, tokenizer) == 0 || counter == size) {
                    printf("Substring: %s\n", substring);
                    printf("Next: %s\n", next);
                    tokens[token_count] = i;
                    printf("Assigned token: %d\n", i);
                    counter = 0;
                    token_count++;

                    memset(substring, '\0', sizeof(&substring));
                    memset(next, '\0', sizeof(&substring));
                    next[counter] = input[c+1];
                };
            }
        }
    }

    free(substring);

    EncodedInput encoded_input;

    encoded_input.input_tokens = tokens;
    encoded_input.num_input_tokens = token_count;

    return encoded_input;
}


void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}
