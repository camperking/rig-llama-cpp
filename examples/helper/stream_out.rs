use rig_agent::agent::{MultiTurnStreamItem, PromptResponse, StreamingResult};
use rig_core::streaming::{Item, StreamEvent};
use tokio_stream::StreamExt;

/// Helper function to stream a completion request to stdout.
pub async fn stream_out(stream: &mut StreamingResult) -> Result<PromptResponse, std::io::Error> {
    let mut final_res = PromptResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Text {
                text,
                ..
            }))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(Item::Event(StreamEvent::Reasoning {
                text,
                ..
            }))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(MultiTurnStreamItem::FinalResponse(res)) => {
                final_res = res;
            }
            Err(err) => {
                eprintln!("Error: {err}");
            }
            _ => {}
        }
    }

    Ok(final_res)
}
