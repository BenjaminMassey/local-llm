mod llama;

fn main() {
    println!("Loading LLM...");
    let mut llm = llama::init();
    println!("LLM loaded.");
    println!("Sending request of \"Tell me a funny joke.\" to loaded LLM...");
    let response = llama::chat(&mut llm, "Tell me a funny joke.", Some(50));
    println!("Response: {response}");
    println!("Sending request of \"Tell me another.\" to loaded LLM...");
    let response = llama::chat(&mut llm, "Tell me another.", Some(50));
    println!("Response: {response}");
    println!("Sending lazy request of \"Tell me a funny joke.\"...");
    let response = llama::lazy_chat("Tell me a funny joke.");
    println!("Response: {response}");
}