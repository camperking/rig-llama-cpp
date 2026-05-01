use rig::completion::CompletionRequest;
use rig::message::{AssistantContent, Message, ToolCall, UserContent};
#[cfg(feature = "mtmd")]
use rig::one_or_many::OneOrMany;
use serde_json::{Value, json};

#[cfg(feature = "mtmd")]
use crate::slot::fnv1a_64;
#[cfg(feature = "mtmd")]
use crate::types::PreparedImage;
use crate::types::PreparedRequest;

/// Normalize a tool result's content list. rig-core 0.35.0's streaming agent
/// loop stored the raw tool-output string as plain `ToolResultContent::Text`
/// in the chat history it sent to the next provider call, while its
/// non-streaming counterpart called `ToolResultContent::from_tool_output` to
/// parse image JSON into `Image` variants. Upstream rig-core 0.36.0 fixed
/// this (see rig PR #1661 / issue #1650), so the streaming and non-streaming
/// paths now agree. This helper is kept as a defensive pass: if a caller
/// hands us a history produced by an older rig-core, or by some other agent
/// that still emits raw Text for image tool outputs, we re-parse here so
/// image content surfaces as `ToolResultContent::Image`. No-op for plain-text
/// outputs: `from_tool_output` falls back to a single Text part on parse
/// failure.
///
/// Only used by mtmd-aware code paths — without `mtmd` enabled, tool images
/// can't be sent to the model anyway.
#[cfg(feature = "mtmd")]
fn normalized_tool_parts(
    content: &OneOrMany<rig::message::ToolResultContent>,
) -> Vec<rig::message::ToolResultContent> {
    let mut out = Vec::new();
    for part in content.iter() {
        match part {
            rig::message::ToolResultContent::Text(t) => {
                let parsed = rig::message::ToolResultContent::from_tool_output(t.text.clone());
                for p in parsed.into_iter() {
                    out.push(p);
                }
            }
            other => out.push(other.clone()),
        }
    }
    out
}

pub(crate) fn prepare_request(request: &CompletionRequest) -> Result<PreparedRequest, String> {
    let mut messages = Vec::new();

    let mut system = request.preamble.clone().unwrap_or_default();
    if let Some(Message::User { content }) = request.normalized_documents() {
        let doc_text: String = content
            .iter()
            .filter_map(|c| match c {
                UserContent::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !doc_text.is_empty() {
            if !system.is_empty() {
                system.push_str("\n\n");
            }
            system.push_str(&doc_text);
        }
    }

    if !system.is_empty() {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }

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
                                "name": tool.name,
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
        Some(rig::message::ToolChoice::Auto) => Some("auto".to_string()),
        Some(rig::message::ToolChoice::None) => Some("none".to_string()),
        Some(rig::message::ToolChoice::Required) => Some("required".to_string()),
        Some(rig::message::ToolChoice::Specific { .. }) => {
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
                            // walk the normalized parts here in the same iteration order.
                            for part in normalized_tool_parts(&tool_result.content) {
                                if let rig::message::ToolResultContent::Image(image) = part {
                                    match extract_image_bytes(&image) {
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
        enable_thinking: request
            .additional_params
            .as_ref()
            .map(has_thinking_request)
            .unwrap_or(false),
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
                                "id": tr.call_id.as_deref().unwrap_or(&tr.id),
                                "type": "function",
                                "function": {
                                    "name": tr.id,
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
                    #[cfg(feature = "mtmd")]
                    let normalized = normalized_tool_parts(&tool_result.content);

                    #[cfg(feature = "mtmd")]
                    let content = normalized
                        .iter()
                        .filter_map(|part| match part {
                            rig::message::ToolResultContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    #[cfg(not(feature = "mtmd"))]
                    let content = tool_result
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            rig::message::ToolResultContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    #[cfg(feature = "mtmd")]
                    let image_count = normalized
                        .iter()
                        .filter(|part| matches!(part, rig::message::ToolResultContent::Image(_)))
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
                        "tool_call_id": tool_result.call_id.as_deref().unwrap_or(&tool_result.id),
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
        Message::Assistant { content, .. } => {
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

fn user_content_text(content: &UserContent) -> Option<String> {
    match content {
        UserContent::Text(text) => Some(text.text.clone()),
        UserContent::Document(document) => Some(document_text(document)),
        _ => None,
    }
}

fn document_text(document: &rig::message::Document) -> String {
    match &document.data {
        rig::message::DocumentSourceKind::String(text)
        | rig::message::DocumentSourceKind::Url(text)
        | rig::message::DocumentSourceKind::Base64(text) => text.clone(),
        rig::message::DocumentSourceKind::Raw(bytes) => String::from_utf8_lossy(bytes).into_owned(),
        rig::message::DocumentSourceKind::Unknown => String::new(),
        _ => String::new(),
    }
}

fn tool_call_json(tool_call: &ToolCall) -> Value {
    // Args may already be a JSON-encoded string (parser fallback for partial output).
    // Re-encoding via `.to_string()` would double-quote it; an invalid string would
    // crash the chat template renderer. Pass valid JSON through, swap the rest for "{}".
    let arguments = match &tool_call.function.arguments {
        Value::String(s) if serde_json::from_str::<Value>(s).is_ok() => s.clone(),
        Value::String(_) => "{}".to_string(),
        other => other.to_string(),
    };
    json!({
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": arguments,
        }
    })
}

#[cfg(feature = "mtmd")]
fn extract_image_bytes(image: &rig::message::Image) -> Result<Vec<u8>, String> {
    use rig::message::DocumentSourceKind;
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

fn has_thinking_request(params: &Value) -> bool {
    // check actual value of reasoning/thinking param if present
    if let Some(reasoning) = params.get("reasoning").or_else(|| params.get("thinking"))
        && let Some(enabled) = reasoning.as_bool()
    {
        return enabled;
    }

    false
}
