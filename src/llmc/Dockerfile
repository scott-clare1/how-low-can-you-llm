# use alpine as base image
FROM ubuntu as build-env
# install build-base meta package inside build-env container
RUN apt-get update && apt-get install -y build-essential
RUN apt-get update && apt-get install -y libgomp1

# change directory to /app
WORKDIR /app
# copy all files from current directory inside the build-env container
COPY . .
# Compile the source code and generate hello binary executable file
RUN make server
# use another container to run the program
FROM ubuntu
# copy binary executable to new container
RUN apt-get update && apt-get install -y gcc g++ libgomp1 make
COPY --from=build-env /app/server /app/server
COPY --from=build-env /app/gpt2_124M.bin /app/gpt2_124M.bin
COPY --from=build-env /app/gpt2_tokenizer.bin /app/gpt2_tokenizer.bin
WORKDIR /app

ENV OMP_NUM_THREADS=4

# at last run the program
CMD ["./server"]
