use serde_json::Value;

use crate::types::{PreparedRequest, PromptBuildResult};

/// Render a [`PreparedRequest`] into a prompt string using the model's built-in
/// chat template, falling back to ChatML.
///
/// `llama-cpp-2` 0.1.147 removed the `apply_chat_template_oaicompat` /
/// `OpenAIChatTemplateParams` path that previously let llama.cpp's jinja
/// engine ingest `tools` / `tool_choice` / `json_schema` directly. The
/// remaining `apply_chat_template` only takes `(role, content)` messages, so
/// tool schemas are injected into the system message here and JSON-schema
/// constraints are applied as a sampler (see `src/sampling.rs`) rather than
/// baked into the prompt.
pub(crate) fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    request: &PreparedRequest,
) -> Result<PromptBuildResult, String> {
    use llama_cpp_2::model::LlamaChatMessage;

    let parsed_messages: Vec<(String, String)> =
        serde_json::from_str::<Vec<Value>>(&request.messages_json)
            .map_err(|e| format!("Message deserialization failed: {e}"))?
            .into_iter()
            .filter_map(|msg| {
                Some((
                    msg.get("role")?.as_str()?.to_string(),
                    message_content_as_text(&msg),
                ))
            })
            .collect();

    // Fold tool schemas into the leading system message when present. The
    // oaicompat path used to hand these to the jinja engine out-of-band; that
    // API is gone, so system-prompt injection is the portable replacement.
    let rendered_messages: Vec<(String, String)> = request
        .tools_json
        .as_deref()
        .map(|tools_json| {
            inject_tools_into_system(&parsed_messages, tools_json, request.tool_choice.as_deref())
        })
        .unwrap_or(parsed_messages.clone());

    // Try the model's built-in chat template.
    let tried_template = model.chat_template(None).ok().and_then(|tmpl| {
        let chat_msgs: Vec<LlamaChatMessage> = rendered_messages
            .iter()
            .map(|(role, content)| LlamaChatMessage::new(role.clone(), content.clone()))
            .collect::<Result<_, _>>()
            .ok()?;
        model.apply_chat_template(&tmpl, &chat_msgs, true).ok()
    });

    if let Some(prompt) = tried_template {
        log::debug!("messages_json: {}", request.messages_json);
        log::debug!("rendered prompt:\n{prompt}");
        return Ok(PromptBuildResult { prompt });
    }

    // Fallback to ChatML.
    let mut prompt = String::new();
    for (role, content) in &rendered_messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(PromptBuildResult { prompt })
}

/// Merge a tool-schema description into the first system message (creating one
/// if none exists), returning a new `(role, content)` vec.
fn inject_tools_into_system(
    messages: &[(String, String)],
    tools_json: &str,
    tool_choice: Option<&str>,
) -> Vec<(String, String)> {
    let tool_directive = build_tool_directive(tools_json, tool_choice);
    let mut out: Vec<(String, String)> = Vec::with_capacity(messages.len());
    let mut injected = false;

    for (role, content) in messages {
        if role == "system" && !injected {
            let merged = if content.trim().is_empty() {
                tool_directive.clone()
            } else {
                format!("{content}\n\n{tool_directive}")
            };
            out.push((role.clone(), merged));
            injected = true;
        } else {
            out.push((role.clone(), content.clone()));
        }
    }

    if !injected {
        out.insert(0, ("system".to_string(), tool_directive));
    }
    out
}

/// Render a portable tool-calling directive describing the available tools and
/// how the model should emit calls. Models trained for OpenAI-style function
/// calling (Qwen, Llama, etc.) generally honour `<tool_call>` JSON blocks
/// described this way.
fn build_tool_directive(tools_json: &str, tool_choice: Option<&str>) -> String {
    let mut directive = String::new();
    directive.push_str("You have access to the following tools:\n");
    directive.push_str(tools_json);
    directive.push_str(
        "\n\nTo call a tool, respond with a JSON object of the form \
         `{\"name\": \"<tool_name>\", \"arguments\": {<key>: <value>, ...}}` \
         wrapped in <tool_call></tool_call> tags. You may emit multiple \
         <tool_call> blocks. Do not place any other text inside the tags.",
    );
    if let Some(choice) = tool_choice
        && choice != "auto"
    {
        let how = match choice {
            "none" => "Do not call any tools for this turn.",
            "required" => "You must call at least one tool in your response.",
            other => other,
        };
        directive.push_str(&format!("\nTool choice: {how}"));
    }
    directive
}

fn message_content_as_text(msg: &Value) -> String {
    match msg.get("content") {
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}
