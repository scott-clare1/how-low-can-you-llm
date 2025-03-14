/*
 This file contains utilities shared between the different training scripts.
 In particular, we define a series of macros xxxCheck that call the corresponding
 C standard library function and check its return code. If an error was reported,
 the program prints some debug information and exits.
*/
#ifndef UTILS_H
#define UTILS_H

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
// implementation of dirent for Windows is in dev/unistd.h
#ifndef _WIN32
#include <dirent.h>
#include <arpa/inet.h>

// ----------------------------------------------------------------------------
// fread convenience utils, with nice handling of error checking using macros
// simple replace fopen, fread, fclose, fseek
// with fopenCheck, freadCheck, fcloseCheck, fseekCheck

extern inline FILE *fopen_check(const char *path, const char *mode, const char *file, int line);

#define fopenCheck(path, mode) fopen_check(path, mode, __FILE__, __LINE__)
extern inline void fread_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line);

#define freadCheck(ptr, size, nmemb, stream) fread_check(ptr, size, nmemb, stream, __FILE__, __LINE__)

extern inline void fclose_check(FILE *fp, const char *file, int line);

#define fcloseCheck(fp) fclose_check(fp, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// malloc error-handling wrapper util

extern inline void *malloc_check(size_t size, const char *file, int line);

#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

extern inline void sclose_check(int sockfd, const char *file, int line);

#define scloseCheck(sockfd) sclose_check(sockfd, __FILE__, __LINE__)

#ifdef _WIN32
extern inline void closesocket_check(int sockfd, const char *file, int line);

#define closesocketCheck(sockfd) closesocket_check(sockfd, __FILE__, __LINE__)
#endif

extern inline void fseek_check(FILE *fp, long off, int whence, const char *file, int line);

#define fseekCheck(fp, off, whence) fseek_check(fp, off, whence, __FILE__, __LINE__)

extern inline void fwrite_check(void *ptr, size_t size, size_t nmemb, FILE *stream, const char *file, int line);

#define fwriteCheck(ptr, size, nmemb, stream) fwrite_check(ptr, size, nmemb, stream, __FILE__, __LINE__)


// ----------------------------------------------------------------------------
// check that all tokens are within range
extern inline void token_check(const int* tokens, int token_count, int vocab_size, const char *file, int line);
#define tokenCheck(tokens, count, vocab) token_check(tokens, count, vocab, __FILE__, __LINE__)

// ----------------------------------------------------------------------------
// I/O ops

extern inline void create_dir_if_not_exists(const char *dir);

extern inline int find_max_step(const char* output_log_dir);

extern inline int ends_with_bin(const char* str);


#endif

#endif
