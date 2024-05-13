use llm::{InferenceParameters, Model};

const LLAMA_PATH: &str = "D:/Development/models-ai/llm/bin/open-llama-13b-open-instruct.ggmlv3.q4_0.bin";
pub struct LLM {
    model: llm::models::Llama,
    session: llm::InferenceSession,
}

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

pub fn init() -> LLM {
    let model = llm::load::<llm::models::Llama>(
        std::path::Path::new(LLAMA_PATH),
        llm::TokenizerSource::Embedded,
        Default::default(),
        |_|{},
    )
    .unwrap_or_else(|err| panic!("Failed to load model: {err}"));
    let session = model.start_session(Default::default());
    LLM { model, session }
}

pub fn chat(llm: &mut LLM, prompt: &str, tokens: Option<usize>) -> String {
    let prompt = &contexted_prompt(prompt);
    let mut response = String::new();
    let _ = llm.session.infer::<std::convert::Infallible>(
        &llm.model,
        &mut rand::thread_rng(),
        &llm::InferenceRequest {
            prompt: llm::Prompt::Text(prompt),
            parameters: &InferenceParameters::default(),
            play_back_previous_tokens: false,
            maximum_token_count: tokens,
        },
        &mut Default::default(),
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
                _ => {},
            };

            Ok(llm::InferenceFeedback::Continue)
        }
    );
    response
}

pub fn lazy_chat(prompt: &str) -> String {
    chat(&mut init(), prompt, None)
}