#![doc(html_root_url = "https://docs.rs/local-llm/0.1.0")]

/*!
This is the `local_llm` crate.
*/

use llama_cpp_rs::options::{ModelOptions, PredictOptions};
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

pub fn init(llama_path: &str) -> LLama {
    let _ = gag::Gag::stderr().unwrap();
    let model_options = ModelOptions {
        n_gpu_layers: if cfg!(feature = "cuda") { 20 } else { 0 },
        context_size: 2048,
        ..Default::default()
    };
    LLama::new(llama_path.into(), &model_options).unwrap()
}

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

pub fn lazy_chat(llama_path: &str, prompt: &str) -> String {
    chat(&mut init(llama_path), prompt, None)
}
