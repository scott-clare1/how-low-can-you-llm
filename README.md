# How Low Can You LLM?

A project aspiring to deploy an LLM to free-tier AWS.

We're starting off on easy-mode, deploying a pre-trained GPT-2 model (it's been through pre-training only don't expect to have a conversation with it).

Thoughts for deploying to AWS with free-tier:
- `t2.micro` 750 hours a month for free. Gives us 1GB memory and 1vCPU.
- 50GB ECR storage.
- Free ECS.

My thoughts are that this should be enough to deploy our server?

## Download Model and Tokenizer

```
cd src/llmc \
./download_starter_pack.sh
```

## Build Server

```
cd src/llmc
```

Build Server Image:

```
docker build -t server . -m 1g
```

Deploy Server:

```
docker run -p 8080:8080 -it server
```

Make Request:

```
curl http://0.0.0.0:8080/
```

## Future Work

- [ ] Deploy services in AWS using IaC.
- [ ] Documentation for GPT-2 Architecture.
- [ ] Train on Tiny Shakespeare dataset.
- [ ] Model Registry.

> [!NOTE]
> This project has cannabalised Andrej Karpathy's [`llm.c`](https://github.com/karpathy/llm.c) project. I've just tried to optimise it as best as possible for inference over training. Take a look at his [implementation](https://github.com/karpathy/llm.c/blob/master/train_gpt2.c).
