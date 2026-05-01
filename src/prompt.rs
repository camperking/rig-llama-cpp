use serde_json::{Value, json};

use crate::types::{PreparedRequest, PromptBuildResult};

pub(crate) fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    request: &PreparedRequest,
) -> Result<PromptBuildResult, String> {
    use llama_cpp_2::model::LlamaChatMessage;
    use llama_cpp_2::openai::OpenAIChatTemplateParams;

    let chat_template_kwargs = json!({ "enable_thinking": request.enable_thinking }).to_string();

    if let Ok(tmpl) = model.chat_template(None) {
        let params = OpenAIChatTemplateParams {
            messages_json: &request.messages_json,
            tools_json: request.tools_json.as_deref(),
            tool_choice: request.tool_choice.as_deref(),
            json_schema: request.json_schema.as_deref(),
            grammar: None,
            reasoning_format: if request.enable_thinking {
                Some("auto")
            } else {
                Some("none")
            },
            chat_template_kwargs: Some(&chat_template_kwargs),
            add_generation_prompt: true,
            use_jinja: true,
            parallel_tool_calls: false,
            enable_thinking: request.enable_thinking,
            add_bos: false,
            add_eos: false,
            parse_tool_calls: request.tools_json.is_some(),
        };

        match model.apply_chat_template_oaicompat(&tmpl, &params) {
            Ok(result) => {
                log::debug!("messages_json: {}", request.messages_json);
                log::debug!("enable_thinking: {}", request.enable_thinking);
                log::debug!("chat_format: {}", result.chat_format);
                log::debug!("has_parser: {}", result.parser.is_some());
                log::debug!(
                    "prompt contains <|think|>: {}",
                    result.prompt.contains("<|think|>")
                );
                log::debug!("rendered prompt:\n{}", result.prompt);
                return Ok(PromptBuildResult {
                    prompt: result.prompt.clone(),
                    template_result: Some(result),
                });
            }
            Err(e) => {
                log::debug!("apply_chat_template_oaicompat failed: {e}, falling back");
                #[cfg(feature = "mtmd")]
                if !request.images.is_empty() {
                    return Err(format!(
                        "Chat template failed for multimodal request: {e}. \
                         The model's chat template may not support the current configuration."
                    ));
                }
            }
        }
    }

    let parsed_messages: Vec<(String, String)> =
        serde_json::from_str::<Vec<Value>>(&request.messages_json)
            .map_err(|e| format!("Message deserialization failed: {e}"))?
            .into_iter()
            .filter_map(|msg| {
                Some((
                    msg.get("role")?.as_str()?.to_string(),
                    message_content_as_text(&msg).to_string(),
                ))
            })
            .collect();

    let chat_msgs: Vec<LlamaChatMessage> = parsed_messages
        .iter()
        .map(|(role, content)| LlamaChatMessage::new(role.clone(), content.clone()))
        .collect::<Result<_, _>>()
        .map_err(|e| format!("Chat message creation failed: {e}"))?;

    // Try model's built-in chat template first
    if let Ok(tmpl) = model.chat_template(None)
        && let Ok(prompt) = model.apply_chat_template(&tmpl, &chat_msgs, true)
    {
        return Ok(PromptBuildResult {
            prompt,
            template_result: None,
        });
    }

    // Fallback to ChatML format
    let mut prompt = String::new();
    for (role, content) in &parsed_messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(PromptBuildResult {
        prompt,
        template_result: None,
    })
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
