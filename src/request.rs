use rig_core::completion::{CompletionRequest, Reasoning};
use rig_core::message::{AssistantContent, Message, ToolCall, ToolResultContent, UserContent};
use serde_json::{Value, json};

#[cfg(feature = "mtmd")]
use crate::slot::fnv1a_64;
#[cfg(feature = "mtmd")]
use crate::types::PreparedImage;
use crate::types::PreparedRequest;

/// Turn a request into the chat-template inputs.
///
/// The request arrives prepared by rig's completion operation: the preamble
/// is the leading `Message::System` of `chat_history`, any documents have
/// joined the history as a user message, and the history is adapted to what
/// [`CompletionWire`](crate::CompletionWire) accepts.
pub(crate) fn prepare_request(request: &CompletionRequest) -> Result<PreparedRequest, String> {
    let mut messages = Vec::new();

    for msg in request.chat_history.iter() {
        append_message_json(&mut messages, msg);
    }

    let tools_json = if request.tools.is_empty() {
        None
    } else {
        Some(
            serde_json::to_string(
                &request
                    .tools
                    .iter()
                    .map(|tool| {
                        json!({
                            "type": "function",
                            "function": {
                                "name": tool.name.as_str(),
                                "description": tool.description,
                                "parameters": tool.parameters,
                            }
                        })
                    })
                    .collect::<Vec<_>>(),
            )
            .map_err(|e| format!("Tool serialization failed: {e}"))?,
        )
    };

    let tool_choice = match request.tool_choice.as_ref() {
        None => None,
        Some(rig_core::message::ToolChoice::Auto) => Some("auto".to_string()),
        Some(rig_core::message::ToolChoice::None) => Some("none".to_string()),
        Some(rig_core::message::ToolChoice::Required) => Some("required".to_string()),
        Some(rig_core::message::ToolChoice::Specific { .. }) => {
            return Err("Specific tool choice is not supported by local llama adapter".into());
        }
    };

    let json_schema = request
        .output_schema
        .as_ref()
        .map(serde_json::to_string)
        .transpose()
        .map_err(|e| format!("Schema serialization failed: {e}"))?;

    #[cfg(feature = "mtmd")]
    let images = {
        let mut imgs = Vec::new();
        for msg in request.chat_history.iter() {
            if let Message::User { content } = msg {
                for item in content.iter() {
                    match item {
                        UserContent::Image(image) => match extract_image_bytes(image) {
                            Ok(bytes) => {
                                let hash = fnv1a_64(&bytes);
                                imgs.push(PreparedImage { bytes, hash });
                            }
                            Err(e) => return Err(format!("Image extraction failed: {e}")),
                        },
                        UserContent::ToolResult(tool_result) => {
                            // Tool results can carry image content (e.g. a `read_file`
                            // tool that reads a `.png`). Bitmap ordering must match the
                            // order media markers appear in `append_message_json`, so we
                            // walk the parts here in the same iteration order.
                            for part in &tool_result.content {
                                if let ToolResultContent::Image(image) = part {
                                    match extract_image_bytes(image) {
                                        Ok(bytes) => {
                                            let hash = fnv1a_64(&bytes);
                                            imgs.push(PreparedImage { bytes, hash });
                                        }
                                        Err(e) => {
                                            return Err(format!(
                                                "Tool-result image extraction failed: {e}"
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        imgs
    };

    Ok(PreparedRequest {
        messages_json: serde_json::to_string(&messages)
            .map_err(|e| format!("Message serialization failed: {e}"))?,
        tools_json,
        tool_choice,
        json_schema,
        enable_thinking: thinking_preference(request),
        #[cfg(feature = "mtmd")]
        images,
    })
}

fn append_message_json(messages: &mut Vec<Value>, msg: &Message) {
    match msg {
        Message::User { content } => {
            #[cfg(feature = "mtmd")]
            let has_images = content
                .iter()
                .any(|item| matches!(item, UserContent::Image(_)));

            #[cfg(feature = "mtmd")]
            if has_images {
                // Use structured content parts matching llama.cpp server behavior.
                // This ensures templates that distinguish media_marker from text
                // (e.g. Qwen3.5-VL) handle images correctly regardless of
                // enable_thinking or reasoning_format settings.
                let mut content_parts = Vec::new();
                for item in content.iter() {
                    match item {
                        UserContent::Image(_) => {
                            content_parts.push(json!({
                                "type": "media_marker",
                                "text": llama_cpp_2::mtmd::mtmd_default_marker()
                            }));
                        }
                        other => {
                            if let Some(text) = user_content_text(other) {
                                content_parts.push(json!({
                                    "type": "text",
                                    "text": text
                                }));
                            }
                        }
                    }
                }
                if !content_parts.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": content_parts,
                    }));
                }
            } else {
                let mut parts = Vec::new();
                for item in content.iter() {
                    if let Some(text) = user_content_text(item) {
                        parts.push(text);
                    }
                }
                let text = parts.join("\n");
                if !text.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": text,
                    }));
                }
            }

            #[cfg(not(feature = "mtmd"))]
            {
                let mut parts = Vec::new();
                for item in content.iter() {
                    if let Some(text) = user_content_text(item) {
                        parts.push(text);
                    }
                }
                let text = parts.join("\n");
                if !text.is_empty() {
                    messages.push(json!({
                        "role": "user",
                        "content": text,
                    }));
                }
            }

            let tool_results: Vec<_> = content
                .iter()
                .filter_map(|c| match c {
                    UserContent::ToolResult(tool_result) => Some(tool_result),
                    _ => None,
                })
                .collect();

            if !tool_results.is_empty() {
                // Some chat templates (e.g. Gemma) require tool results to be preceded
                // by an assistant message with matching tool_calls. Rig's agent loop
                // may not always include this, so synthesize one when missing.
                let has_preceding_tool_calls = messages
                    .last()
                    .and_then(|m| m.get("tool_calls"))
                    .and_then(Value::as_array)
                    .is_some_and(|arr| !arr.is_empty());

                if !has_preceding_tool_calls {
                    let synthetic_tool_calls: Vec<Value> = tool_results
                        .iter()
                        .map(|tr| {
                            json!({
                                "id": tr.call.wire(),
                                "type": "function",
                                "function": {
                                    "name": tr.name.as_str(),
                                    "arguments": "{}",
                                }
                            })
                        })
                        .collect();

                    messages.push(json!({
                        "role": "assistant",
                        "content": Value::Null,
                        "tool_calls": synthetic_tool_calls,
                    }));
                }

                #[cfg(feature = "mtmd")]
                let mut pending_tool_image_count: usize = 0;

                for tool_result in tool_results {
                    let content = tool_result
                        .content
                        .iter()
                        .filter_map(tool_result_text)
                        .collect::<Vec<_>>()
                        .join("\n");

                    #[cfg(feature = "mtmd")]
                    let image_count = tool_result
                        .content
                        .iter()
                        .filter(|part| matches!(part, ToolResultContent::Image(_)))
                        .count();

                    #[cfg(feature = "mtmd")]
                    let final_content = if image_count > 0 && content.is_empty() {
                        // OAI-compat tool messages must be a non-empty string;
                        // an empty content for `role: "tool"` makes some chat
                        // templates emit a malformed turn. Drop a brief
                        // placeholder so the model knows the call returned and
                        // expects the image to follow.
                        format!("[returned {image_count} image(s); see next message]")
                    } else {
                        content
                    };
                    #[cfg(not(feature = "mtmd"))]
                    let final_content = content;

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_result.call.wire(),
                        "content": final_content,
                    }));

                    #[cfg(feature = "mtmd")]
                    {
                        pending_tool_image_count += image_count;
                    }
                }

                // Tool-result images can't ride along with `role: "tool"` content
                // in llama.cpp's OAI-compat chat template (multimodal markers are
                // only honored in user messages). Emit a synthetic user message
                // carrying one media_marker per tool-result image so the bitmaps
                // collected in `prepare_request` line up positionally with the
                // markers in the rendered prompt.
                #[cfg(feature = "mtmd")]
                if pending_tool_image_count > 0 {
                    let mut content_parts: Vec<Value> =
                        Vec::with_capacity(pending_tool_image_count + 1);
                    content_parts.push(json!({
                        "type": "text",
                        "text": "Image(s) returned by the tool call above:",
                    }));
                    for _ in 0..pending_tool_image_count {
                        content_parts.push(json!({
                            "type": "media_marker",
                            "text": llama_cpp_2::mtmd::mtmd_default_marker(),
                        }));
                    }
                    messages.push(json!({
                        "role": "user",
                        "content": content_parts,
                    }));
                }
            }
        }
        Message::Assistant(assistant) => {
            let content = &assistant.content;
            let text = content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::Text(t) => Some(t.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");

            let tool_calls = content
                .iter()
                .filter_map(|c| match c {
                    AssistantContent::ToolCall(tool_call) => Some(tool_call),
                    _ => None,
                })
                .map(tool_call_json)
                .collect::<Vec<_>>();

            if !text.is_empty() || !tool_calls.is_empty() {
                messages.push(json!({
                    "role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { Value::String(text) },
                    "tool_calls": if tool_calls.is_empty() { Value::Null } else { Value::Array(tool_calls) },
                }));
            }
        }
        Message::System { content } => {
            messages.push(json!({
                "role": "system",
                "content": content,
            }));
        }
    }
}

/// The text a tool result part contributes to a `role: "tool"` message.
/// Structured JSON is sent as its JSON text; images travel separately.
fn tool_result_text(part: &ToolResultContent) -> Option<String> {
    match part {
        ToolResultContent::Text(text) => Some(text.text.clone()),
        ToolResultContent::Json { value } => Some(value.to_string()),
        ToolResultContent::Image(_) => None,
    }
}

fn user_content_text(content: &UserContent) -> Option<String> {
    match content {
        UserContent::Text(text) => Some(text.text.clone()),
        UserContent::Document(document) => Some(document_text(document)),
        _ => None,
    }
}

fn document_text(document: &rig_core::message::Document) -> String {
    match &document.data {
        rig_core::message::DocumentSourceKind::String(text)
        | rig_core::message::DocumentSourceKind::Url(text)
        | rig_core::message::DocumentSourceKind::Base64(text)
        | rig_core::message::DocumentSourceKind::FileId(text) => text.clone(),
        rig_core::message::DocumentSourceKind::Raw(bytes) => {
            String::from_utf8_lossy(bytes).into_owned()
        }
        rig_core::message::DocumentSourceKind::Unknown => String::new(),
    }
}

fn tool_call_json(tool_call: &ToolCall) -> Value {
    // rig keeps arguments as a JSON object (malformed ones arrive as `{}`), so
    // the encoded text is always valid JSON for the chat template to render.
    let arguments = Value::Object(tool_call.function.arguments.clone()).to_string();
    json!({
        "id": tool_call.id.wire(),
        "type": "function",
        "function": {
            "name": tool_call.function.name.as_str(),
            "arguments": arguments,
        }
    })
}

#[cfg(feature = "mtmd")]
fn extract_image_bytes(image: &rig_core::message::Image) -> Result<Vec<u8>, String> {
    use rig_core::message::DocumentSourceKind;
    match &image.data {
        DocumentSourceKind::Raw(bytes) => Ok(bytes.clone()),
        DocumentSourceKind::Base64(encoded) => {
            use base64::Engine;
            base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|e| format!("Base64 decode failed: {e}"))
        }
        DocumentSourceKind::Url(_) => {
            Err("URL image sources are not supported; pre-fetch the image data".into())
        }
        other => Err(format!("Unsupported image source kind: {other:?}")),
    }
}

/// The caller's reasoning preference, or `None` if they expressed none.
///
/// Read from the portable `reasoning` option first: `Reasoning::Off` turns
/// thinking off and any effort or budget turns it on. A boolean `reasoning`
/// or `thinking` key in `additional_params` is still honoured when the
/// option is unset.
///
/// The distinction matters: a template that gates reasoning on
/// `enable_thinking is defined` reads an unconditional `false` as "thinking
/// off", which is not what an absent parameter means. See
/// [`crate::types::PreparedRequest::enable_thinking`].
fn thinking_preference(request: &CompletionRequest) -> Option<bool> {
    if let Some(reasoning) = &request.options.reasoning {
        return Some(!matches!(reasoning, Reasoning::Off));
    }
    let params = request.additional_params.as_ref()?;
    params
        .get("reasoning")
        .or_else(|| params.get("thinking"))
        .and_then(Value::as_bool)
}

#[cfg(test)]
mod tests {
    use rig_core::completion::{CompletionRequest, Effort, Reasoning};
    use rig_core::message::{Message, ToolCall, ToolFunction, ToolName, ToolResultContent};
    use serde_json::{Value, json};

    use super::prepare_request;

    fn messages(request: &CompletionRequest) -> Vec<Value> {
        let prepared = prepare_request(request).unwrap();
        serde_json::from_str(&prepared.messages_json).unwrap()
    }

    #[test]
    fn the_preamble_renders_as_the_leading_system_message() {
        let request = CompletionRequest::new("hi").preamble("Be brief.");
        let messages = messages(&request);
        assert_eq!(
            messages[0],
            json!({"role": "system", "content": "Be brief."})
        );
        assert_eq!(messages[1], json!({"role": "user", "content": "hi"}));
    }

    #[test]
    fn a_tool_result_answers_the_call_it_names() {
        let call = ToolCall::from_wire(
            "",
            ToolFunction::new(ToolName::new("get_time").unwrap(), json!({"zone": "UTC"})),
        );
        let result = Message::from(call.result(vec![ToolResultContent::text("12:00")]));
        let request = CompletionRequest::new("Continue.").messages([
            Message::user("What time is it?"),
            Message::from(call.clone()),
            result,
        ]);
        let messages = messages(&request);

        let id = call.id.wire().into_owned();
        let tool_call = &messages[1]["tool_calls"][0];
        assert_eq!(tool_call["id"], json!(id));
        assert_eq!(tool_call["function"]["name"], "get_time");
        // Templates expect the arguments as JSON text.
        assert_eq!(tool_call["function"]["arguments"], "{\"zone\":\"UTC\"}");
        assert_eq!(
            messages[2],
            json!({"role": "tool", "tool_call_id": id, "content": "12:00"})
        );
    }

    #[test]
    fn the_reasoning_option_sets_the_thinking_preference() {
        let prepare = |request: CompletionRequest| prepare_request(&request).unwrap();
        assert_eq!(prepare(CompletionRequest::new("hi")).enable_thinking, None);
        assert_eq!(
            prepare(CompletionRequest::new("hi").reasoning(Reasoning::Off)).enable_thinking,
            Some(false)
        );
        assert_eq!(
            prepare(CompletionRequest::new("hi").reasoning(Effort::High)).enable_thinking,
            Some(true)
        );
        // The pre-0.6 `additional_params` switch still works.
        assert_eq!(
            prepare(CompletionRequest::new("hi").additional_params(json!({"thinking": false})))
                .enable_thinking,
            Some(false)
        );
    }
}
