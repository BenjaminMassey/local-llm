pub use llama_cpp_rs::LLama;
use llama_cpp_rs::options::{ModelOptions, PredictOptions};

fn contexted_prompt(query: &str) -> String {
    format!(
r#"<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{{ you are a helpful assistant }}<|eot_id|><|start_header_id|>user<|end_header_id|>
{{ {query} }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"#
    )
}

fn trim_trailing_newlines(string: &str) -> String {
    let mut s = string.to_owned();
    while s.chars().last() == Some('\n') {
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
        tokens: if tokens.is_none() { 0 } else { tokens.unwrap() as i32 },
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
    let result = llama
        .predict(
            prompt.into(),
            predict_options,
        )
        .unwrap();
    trim_trailing_newlines(&result)
}

pub fn lazy_chat(llama_path: &str, prompt: &str) -> String {
    chat(&mut init(llama_path), prompt, None)
}