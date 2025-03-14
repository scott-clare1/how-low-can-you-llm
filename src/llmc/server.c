#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>
#include "inference.h"
#include "tokenizer.h"

#define PORT 8080

int main(int argc, char const* argv[]) {
    int server_fd, new_socket, valread;

    int T = 128;
    uint64_t rng_state = 1337;

    // build the Tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    struct sockaddr_in address;

    int opt = 1;

    socklen_t addrlen = sizeof(address);

    printf("Creating socket\n");
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    printf("Created socket: %d\n", server_fd);

    if (server_fd < 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }

    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(PORT);

    printf("Binding to port: %d\n", PORT);
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind");
        exit(EXIT_FAILURE);
    }

    while (1) {
        printf("Building model\n");
        GPT2 model;
        gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");

        if(model.acts_memory == NULL) {
            // record the current B,T as well
            model.seq_len = T;
            // and now allocate the space
            fill_in_activation_sizes(model.act_sizes, model.config, T);

            model.acts_memory = malloc_and_point_activations(&model.acts, model.act_sizes);
            // also create memory for caching inputs and targets
            model.inputs = (int*)mallocCheck(T * sizeof(int));
        } else {
            // validate B,T is consistent with how we've allocated the memory before
            // in principle we could get more clever here in the future, for now this is safest
            if (T != model.seq_len) {
                printf("Model: T=%d, Desired: T=%d\n", model.seq_len, (int)T);
                exit(EXIT_FAILURE);
            }
        }

        ActivationTensors acts = model.acts;

        size_t L = model.config.num_layers;
        size_t C = model.config.channels;

        printf("Listening for connections.\n");
        if (listen(server_fd, 3) < 0) {
            perror("listen");
            exit(EXIT_FAILURE);
        }

        printf("Creating new socket to deal with client request.\n");
        new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen);
        printf("Created socket: %d\n", new_socket);
        if (new_socket < 0) {
            perror("accept");
            exit(EXIT_FAILURE);
        }

        printf("Connection established, ready to receive data.\n");

        int* gen_tokens = (int*)mallocCheck(T * sizeof(uint32_t));

        gen_tokens[0] = tokenizer.eot_token;

        char buffer[1024] = {};

        for (int t = 1; t < T; t++) {
            const char* token = get_next_token(gen_tokens, &model, &rng_state, T, tokenizer, t);
            strcat(buffer, token);
        }

        char response[1024];

        int response_length = snprintf(response, sizeof(response),
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/plain\r\n"
            "Content-Length: %ld\r\n"
            "Connection: close\r\n"
            "\r\n"
            "%s",
            strlen(buffer), buffer);

        send(new_socket, response, response_length, 0);
        free(gen_tokens);
        gpt2_free(&model);
    }

    printf("Closing connection\n");

    close(new_socket);
    close(server_fd);

    return 0;
}
