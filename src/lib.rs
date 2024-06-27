#![doc(html_root_url = "https://docs.rs/local-llm/0.1.0")]

//! `local_llm`:  a high level wrapper for llama.cpp bindings
//!
//! See [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) and [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp) for deeper insight. 
//!
//! The primary objective of this project is to provide a more streamlined way to load
//! an large language model (LLM) from a model file and then easily "chat" with it. This
//! is in response to most localized LLM crates being much lower level, and often only
//! including basic inference, rather than any processing that provides a chat-like interaction.
//!
//! # Example
//! 
//! ```
//! let mut llama = local_llm::init("C:/models/llama-model.gguf");
//! let prompt = "What steps would I take to write a crate in Rust?";
//! let response = local_llm::chat(
//!     &mut llama,
//!     prompt,
//!     Some(50),
//! );
//! println!("Prompt:\n{prompt}\n\nResponse:\n{response}");
//! ```
//!
//! # Features
//! 
//! The only current feature is `cuda`, which primarily  enables the `cuda` feature for the
//! [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp) crate. This provides
//! usage of the GPU via CUDA, which will need to be installed separated from
//! [NVIDIA's download website](https://developer.nvidia.com/cuda-downloads).
//! 
//! # Current limitations
//! 
//! For the most part, this crate is directly using the Rust bindings to
//! [mdrokz/rust-llama.cpp](https://github.com/mdrokz/rust-llama.cpp), which is directly
//! calling upon functionality of [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp),
//! so limitations upon direct calls to those are present.
//! 
//! ### Performance
//! The most damaging limitation is performance: running the same model file in a program like
//! [GPT4All](https://gpt4all.io/index.html) appears to be about 40x faster in terms of tokens
//! per second. Inference and model settings were based upon [GPT4All](https://gpt4all.io/index.html),
//! so I cannot currently explain this other than a performance difference between
//! [llama.cpp](https://github.com/ggerganov/llama.cpp) and whatever LLM backend
//! [GPT4All](https://gpt4all.io/index.html) is using. This assessment is still TBD, and further
//! work on performance is a WIP.
//! 
//! ### Tokens
//! Another limitation is that token count setting simply harshly cuts off the output of
//! a response, rather than having the model specifically work towards some token count.
//! This means that it can be useful for something like 
//! ```
//! let mut llama = local_llm::init("C:/models/llama-model.gguf");
//! let sentence = "The cookie was eaten.";
//! let prompt = format!(
//!     "Is there an 'X' in the following sentence? Respond only with 'yes' or 'no'.
//!      Here's the sentence: {sentence}"
//! );
//! let response = local_llm::chat(
//!     &mut llama,
//!     &prompt,
//!     Some(1),
//! );
//! let contains_x = response.to_lowercase().contains("yes");
//! ```
//! where one knows that they are happy to cut off after one word, but is not useful for
//! something like
//! ```
//! let mut llama = local_llm::init("C:/models/llama-model.gguf");
//! let response = local_llm::chat(
//!     &mut llama,
//!     "Write me an essay on the American revolution.",
//!     Some(150),
//! );
//! ```
//! where one wants a particular size answer (150 tokens), since the LLM will instead
//! start to craft a response of some determined size, but then arbitrarily cut off
//! after the token count (rather than craft the response with the size in mind).
//! 
//! ### Settings
//! The last major limitation present is machine-specific settings. There is an intellectual
//! conflict in the setup of this crate. On one hand, the aim of the crate is to provide
//! as easy-to-use of an LLM interface as possible, so it seeks to set up all settings
//! for the user itself. On the other hand, there seems to always be _some_ amount of
//! settings that are usage and machine dependent such that this is an issue of an idea.
//! In particular, the crate is currently set to load 20 GPU layers for a CUDA-backed
//! LLM model. This number will actually depend on a combination of desired GPU usage
//! plus available GPU power, so a hard-coded value is undesirable, but a required setting
//! value also goes against the ideology of this high level crate. This will be tackled
//! through a settings system and/or CUDA communcation basis in the future, but is currently
//! a limitation and WIP.

use llama_cpp_rs::options::{ModelOptions, PredictOptions};
/// An LLM instance for and from the `llama_cpp_rs` crate.
pub use llama_cpp_rs::LLama;

fn contexted_prompt(query: &str) -> String {
    format!(
        r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ you are a helpful assistant }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ {query} }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"#
    )
}

fn trim_trailing_newlines(string: &str) -> String {
    let mut s = string.to_owned();
    while s.ends_with('\n') {
        s.pop();
    }
    s
}

/// Load an LLM model file (GGUF).
///
/// This function loads a .gguf model file to be used by llama.cpp's LLM backend
/// as the model running LLM inference. It will return a llama_cpp_rs::LLama instance
/// (accessible via this crate), which can be passed to the `chat(..)` function.
///
/// ```
/// let mut llama: local_llm::LLama = local_llm::init("/path/to/model.gguf");
/// ```
/// 
/// Note that this is embedded for one-off instances in the `lazy_chat(..)` function.
pub fn init(llama_path: &str) -> LLama {
    let _ = gag::Gag::stderr().unwrap();
    let model_options = ModelOptions {
        n_gpu_layers: if cfg!(feature = "cuda") { 20 } else { 0 },
        context_size: 2048,
        ..Default::default()
    };
    LLama::new(llama_path.into(), &model_options).unwrap()
}

/// Chat request with an LLM instance.
///
/// This function takes a llama_cpp_rs::LLama instance `llama`, runs inference with
/// a chat-style wrapped `prompt`, optionally cuts off inference at some `tokens` amount,
/// and returns the String response.
///
/// ```
/// let response: String = local_llm::chat(
///     &mut local_llm::init("/path/to/model.gguf"),
///     "How do I write a resume?",
///     Some(100),
/// );
/// ```
/// 
/// Note that this is embedded for one-off instances in the `lazy_chat(..)` function.
pub fn chat(llama: &mut LLama, prompt: &str, tokens: Option<usize>) -> String {
    let prompt = &contexted_prompt(prompt);
    let predict_options = PredictOptions {
        tokens: if let Some(t) = tokens { t as i32 } else { 0 },
        token_callback: Some(Box::new(|token| {
            !(token.contains("===") || token.contains("<|"))
            // TODO: none of this seems proper, and it seems to break itself out
            //       rather reasonably without this, so unsure
        })),
        repeat: 64,
        penalty: 1.3,
        temperature: 0.1,
        top_k: 40,
        top_p: 0.95,
        threads: 8,
        ..Default::default()
    };
    let result = llama.predict(prompt.into(), predict_options).unwrap();
    trim_trailing_newlines(&result)
}

/// Quick LLM chatting for one-off instances.
///
/// This function loads an LLM model from a `llama_path` .gguf file using
/// the `init(..)` function, passes that model instance and a `prompt`
/// string to the `chat(..)` function, and returns the `chat(..)` function's
/// response. It gives no token cutoff amount.
///
/// ```
/// let response: String = local_llm::lazy_chat(
///     &mut local_llm::init("/path/to/model.gguf"),
///     "How do I write a resume?",
/// );
/// ```
/// 
/// See `init(..)` and `chat(..)`, since this simply wraps them.
pub fn lazy_chat(llama_path: &str, prompt: &str) -> String {
    chat(&mut init(llama_path), prompt, None)
}
