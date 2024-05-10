mod llama;

fn main() {
    println!("Sending request...");
    let response = llama::chat("Tell me a funny joke.", 50);
    println!("Response: {response}");
}