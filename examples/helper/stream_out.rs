use rig::agent::{FinalResponse, MultiTurnStreamItem, StreamingResult, Text};
use rig::streaming::StreamedAssistantContent;
use tokio_stream::StreamExt;

/// Helper function to stream a completion request to stdout.
pub async fn stream_out<R>(
    stream: &mut StreamingResult<R>,
) -> Result<FinalResponse, std::io::Error> {
    let mut final_res = FinalResponse::empty();
    print!("Response: ");
    while let Some(content) = stream.next().await {
        match content {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamedAssistantContent::Text(
                Text { text },
            ))) => {
                print!("{text}");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            Ok(MultiTurnStreamItem::StreamAssistantItem(
                StreamedAssistantContent::ReasoningDelta { id: _, reasoning },
            )) => {
                // let reasoning = display_text(reasoning);
                print!("{reasoning}");
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
