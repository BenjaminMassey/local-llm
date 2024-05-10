mod llama;

fn main() {
    println!("Starting LLM...");
    let mut llm = llama::init();
    println!("Sending request...");
    let response = llama::chat(&mut llm, "Tell me a funny joke.", Some(50));
    println!("Response: {response}");
    println!("Sending request...");
    let response = llama::chat(&mut llm, "Tell me another.", Some(50));
    println!("Response: {response}");
    let response = llama::lazy_chat("Tell me a funny joke.");
    println!("Response: {response}");
}