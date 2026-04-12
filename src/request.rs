use rig::completion::CompletionRequest;
use rig::message::{AssistantContent, Message, ToolCall, UserContent};
use serde_json::{Value, json};

use crate::types::PreparedRequest;

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
                    if let UserContent::Image(image) = item {
                        match extract_image_bytes(image) {
                            Ok(bytes) => imgs.push(bytes),
                            Err(e) => return Err(format!("Image extraction failed: {e}")),
                        }
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

                for tool_result in tool_results {
                    let content = tool_result
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            rig::message::ToolResultContent::Text(text) => {
                                Some(text.text.as_str())
                            }
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_result.call_id.as_deref().unwrap_or(&tool_result.id),
                        "content": content,
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
    json!({
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments.to_string(),
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
