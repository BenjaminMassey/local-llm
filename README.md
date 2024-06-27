![Maintenance](https://img.shields.io/badge/maintenance-actively--developed-brightgreen.svg)
[![crates-io](https://img.shields.io/crates/v/local-llm.svg)](https://crates.io/crates/local-llm)
[![api-docs](https://docs.rs/local-llm/badge.svg)](https://docs.rs/local-llm)
[![dependency-status](https://deps.rs/repo/github/BenjaminMassey/local-llm/status.svg)](https://deps.rs/repo/github/BenjaminMassey/local-llm)

# local-llm
Copyright &copy; 2024 Benjamin Massey (Version 0.1.0)

A simplified wrapper around the "llama_cpp_rs" crate for local usage of a Llama LLM.

`local_llm`:  a high level wrapper for llama.cpp bindings

See [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) and [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp) for deeper insight.

The primary objective of this project is to provide a more streamlined way to load
an large language model (LLM) from a model file and then easily "chat" with it. This
is in response to most localized LLM crates being much lower level, and often only
including basic inference, rather than any processing that provides a chat-like interaction.

## Example

```rust
let mut llama = local_llm::init("C:/models/llama-model.gguf");
let prompt = "What steps would I take to write a crate in Rust?";
let response = local_llm::chat(
    &mut llama,
    prompt,
    Some(50),
);
println!("Prompt:\n{prompt}\n\nResponse:\n{response}");
```

## Features

The only current feature is `cuda`, which primarily  enables the `cuda` feature for the
[mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp) crate. This provides
usage of the GPU via CUDA, which will need to be installed separated from
[NVIDIA's download website](https://developer.nvidia.com/cuda-downloads).

## Current limitations

For the most part, this crate is directly using the Rust bindings to
[mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp), which is directly
calling upon functionality of [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp),
so limitations upon direct calls to those are present.

#### Performance
The most damaging limitation is performance: running the same model file in a program like
[GPT4All](https://gpt4all.io/index.html) appears to be about 40x faster in terms of tokens
per second. Inference and model settings were based upon [GPT4All](https://gpt4all.io/index.html),
so I cannot currently explain this other than a performance difference between
[llama.cpp](https://github.com/ggerganov/llama.cpp) and whatever LLM backend
[GPT4All](https://gpt4all.io/index.html) is using. This assessment is still TBD, and further
work on performance is a WIP.

#### Tokens
Another limitation is that token count setting simply harshly cuts off the output of
a response, rather than having the model specifically work towards some token count.
This means that it can be useful for something like
```rust
let mut llama = local_llm::init("C:/models/llama-model.gguf");
let sentence = "The cookie was eaten.";
let prompt = format!(
    "Is there an 'X' in the following sentence? Respond only with 'yes' or 'no'.
     Here's the sentence: {sentence}"
);
let response = local_llm::chat(
    &mut llama,
    &prompt,
    Some(1),
);
let contains_x = response.to_lowercase().contains("yes");
```
where one knows that they are happy to cut off after one word, but is not useful for
something like
```rust
let mut llama = local_llm::init("C:/models/llama-model.gguf");
let response = local_llm::chat(
    &mut llama,
    "Write me an essay on the American revolution.",
    Some(150),
);
```
where one wants a particular size answer (150 tokens), since the LLM will instead
start to craft a response of some determined size, but then arbitrarily cut off
after the token count (rather than craft the response with the size in mind).

#### Settings
The last major limitation present is machine-specific settings. There is an intellectual
conflict in the setup of this crate. On one hand, the aim of the crate is to provide
as easy-to-use of an LLM interface as possible, so it seeks to set up all settings
for the user itself. On the other hand, there seems to always be _some_ amount of
settings that are usage and machine dependent such that this is an issue of an idea.
In particular, the crate is currently set to load 20 GPU layers for a CUDA-backed
LLM model. This number will actually depend on a combination of desired GPU usage
plus available GPU power, so a hard-coded value is undesirable, but a required setting
value also goes against the ideology of this high level crate. This will be tackled
through a settings system and/or CUDA communcation basis in the future, but is currently
a limitation and WIP.

# License

This work is licensed under the "[MIT License](https://opensource.org/license/mit)".
