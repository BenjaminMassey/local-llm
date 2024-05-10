use llm::{InferenceParameters, Model};

fn contexted_prompt(query: &str) -> String {
    format!(
r#"### System:
You are an AI assistant who gives a quality response to whatever humans ask of you.

### Human:
{query}

### Assistant:
"#)
}

fn last_n_chars(string: &str, n: usize) -> String {
    if string.len() <= n {
        return string.to_owned();
    }
    string[
        string
            .char_indices()
            .nth_back(n - 1)
            .unwrap().0..
    ].to_owned()
}

pub fn chat(prompt: &str, tokens: usize) -> String {
    // load a GGML model from disk
    let llama = llm::load::<llm::models::Llama>(
        std::path::Path::new("D:/Development/models-ai/llm/bin/open-llama-13b-open-instruct.ggmlv3.q4_0.bin"),
        llm::TokenizerSource::Embedded,
        Default::default(),
        |_|{},
    )
    .unwrap_or_else(|err| panic!("Failed to load model: {err}"));

    // use the model to generate text from a prompt
    let mut session = llama.start_session(Default::default());
    let prompt = &contexted_prompt(prompt);
    let mut response = String::new();
    let _ = session.infer::<std::convert::Infallible>(
        // model to use for text generation
        &llama,
        // randomness provider
        &mut rand::thread_rng(),
        // the prompt to use for text generation, as well as other
        // inference parameters
        &llm::InferenceRequest {
            prompt: llm::Prompt::Text(prompt),
            parameters: &InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: Some(tokens),
        },
        // llm::OutputRequest
        &mut Default::default(),
        // output callback
        |t| { 
            match t {
                llm::InferenceResponse::InferredToken(x) => {
                    if last_n_chars(&response, 3) == "###" {
                        return Ok(llm::InferenceFeedback::Halt);
                    }
                    response = response.clone() + &x;
                },
                llm::InferenceResponse::EotToken => {
                    return Ok(llm::InferenceFeedback::Halt);
                },
                _ => {}
            };

            Ok(llm::InferenceFeedback::Continue)
        }
    );

    response
}