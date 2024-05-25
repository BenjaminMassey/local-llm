use llama_cpp_rs::{
    options::{ModelOptions, PredictOptions},
    LLama,
};

fn contexted_prompt(query: &str) -> String {
    format!(
r#"# System:
You are an AI assistant who gives a quality response to whatever humans ask of you.

# Human:
{query}

# Assistant:
"#)
}

pub fn init(llama_path: &str) -> LLama {
    let model_options = ModelOptions {
        n_gpu_layers: 30,
        ..Default::default()
    };
    LLama::new(llama_path.into(), &model_options).unwrap()
}

pub fn chat(llama: &mut LLama, prompt: &str, tokens: Option<usize>) -> String {
    let prompt = &contexted_prompt(prompt);
    let predict_options = PredictOptions {
        tokens: if tokens.is_none() { 0 } else { tokens.unwrap() as i32 },
        token_callback: Some(Box::new(|token| {
            token != "#"
        })),
        ..Default::default()
    };
    llama
        .predict(
            prompt.into(),
            predict_options,
        )
        .unwrap()
}

pub fn lazy_chat(llama_path: &str, prompt: &str) -> String {
    chat(&mut init(llama_path), prompt, None)
}