use rig::message::{AssistantContent, Reasoning, ToolCall, ToolFunction};
use rig::one_or_many::OneOrMany;
use rig::streaming::{RawStreamingChoice, RawStreamingToolCall, ToolCallDeltaContent};
use serde_json::Value;

use crate::types::{StreamChunk, StreamDeltaState};

impl StreamDeltaState {
    pub(crate) fn new() -> Self {
        Self {
            tool_calls: std::collections::HashMap::new(),
        }
    }

    pub(crate) fn parse_delta(
        &mut self,
        delta_json: &str,
    ) -> Vec<RawStreamingChoice<StreamChunk>> {
        let mut choices = Vec::new();
        let Ok(value) = serde_json::from_str::<Value>(delta_json) else {
            return choices;
        };
        let Some(obj) = value.as_object() else {
            return choices;
        };

        if let Some(content) = obj.get("content").and_then(Value::as_str)
            && !content.is_empty()
        {
            choices.push(RawStreamingChoice::Message(content.to_string()));
        }

        if let Some(reasoning) = obj.get("reasoning_content").and_then(Value::as_str)
            && !reasoning.is_empty()
        {
            choices.push(RawStreamingChoice::ReasoningDelta {
                id: None,
                reasoning: reasoning.to_string(),
            });
        }

        if let Some(tool_calls) = obj.get("tool_calls").and_then(Value::as_array) {
            for tc in tool_calls {
                let index = tc.get("index").and_then(Value::as_u64).unwrap_or(0);

                // Get or create the accumulated tool call entry.
                // RawStreamingToolCall::empty() generates a unique internal_call_id via nanoid.
                let existing = self
                    .tool_calls
                    .entry(index)
                    .or_insert_with(RawStreamingToolCall::empty);

                // First delta carries the provider-supplied id
                if let Some(id) = tc.get("id").and_then(Value::as_str)
                    && !id.is_empty()
                {
                    existing.id = id.to_string();
                }

                if let Some(function) = tc.get("function").and_then(Value::as_object) {
                    if let Some(name) = function.get("name").and_then(Value::as_str)
                        && !name.is_empty()
                    {
                        existing.name = name.to_string();

                        choices.push(RawStreamingChoice::ToolCallDelta {
                            id: existing.id.clone(),
                            internal_call_id: existing.internal_call_id.clone(),
                            content: ToolCallDeltaContent::Name(name.to_string()),
                        });
                    }
                    if let Some(arguments) = function.get("arguments").and_then(Value::as_str)
                        && !arguments.is_empty()
                    {
                        // Accumulate arguments like the OpenAI implementation
                        let current_args = match &existing.arguments {
                            Value::Null => String::new(),
                            Value::String(s) => s.clone(),
                            v => v.to_string(),
                        };
                        let combined = format!("{current_args}{arguments}");
                        if combined.trim_start().starts_with('{')
                            && combined.trim_end().ends_with('}')
                        {
                            match serde_json::from_str(&combined) {
                                Ok(parsed) => existing.arguments = parsed,
                                Err(_) => existing.arguments = Value::String(combined),
                            }
                        } else {
                            existing.arguments = Value::String(combined);
                        }

                        choices.push(RawStreamingChoice::ToolCallDelta {
                            id: existing.id.clone(),
                            internal_call_id: existing.internal_call_id.clone(),
                            content: ToolCallDeltaContent::Delta(arguments.to_string()),
                        });
                    }
                }
            }
        }

        choices
    }

    /// Flush all accumulated tool calls as complete RawStreamingChoice::ToolCall events.
    ///
    /// If any tool call has incomplete arguments (a `Value::String` that doesn't parse
    /// as a JSON object), we re-parse from the complete `output` using the chat template's
    /// `parse_response_oaicompat`, which reliably extracts tool calls from the full text.
    pub(crate) fn flush_tool_calls(
        &mut self,
        output: &str,
        template_result: Option<&llama_cpp_2::model::ChatTemplateResult>,
    ) -> Vec<RawStreamingChoice<StreamChunk>> {
        let mut tool_calls: Vec<(u64, RawStreamingToolCall)> = self
            .tool_calls
            .drain()
            .filter(|(_, tc)| !tc.name.is_empty())
            .collect();

        // Check if any tool call has broken arguments
        let has_broken = tool_calls
            .iter()
            .any(|(_, tc)| !is_valid_json_args(&tc.arguments));

        if has_broken
            && let Some(reparsed) = reparse_tool_calls_from_output(output, template_result)
        {
            for (_, tc) in &mut tool_calls {
                if !is_valid_json_args(&tc.arguments) {
                    // Find a matching tool call by name in the reparsed set
                    if let Some(fixed_args) = reparsed
                        .iter()
                        .find(|(name, _)| name == &tc.name)
                        .map(|(_, args)| args.clone())
                    {
                        tc.arguments = fixed_args;
                    }
                }
            }
        }

        tool_calls
            .into_iter()
            .map(|(_, tool_call)| RawStreamingChoice::ToolCall(tool_call))
            .collect()
    }
}

/// Returns true if the arguments represent valid JSON (an object or a string that parses as one).
fn is_valid_json_args(args: &Value) -> bool {
    match args {
        Value::Object(_) => true,
        Value::String(s) => serde_json::from_str::<Value>(s)
            .ok()
            .is_some_and(|v| v.is_object()),
        Value::Null => true, // no-arg tool calls are fine
        _ => false,
    }
}

/// Re-parse tool calls from the complete output using the chat template parser.
/// Falls back to manual XML parsing for models that emit `<tool_call>` XML format.
/// Returns a list of (name, arguments) pairs on success.
fn reparse_tool_calls_from_output(
    output: &str,
    template_result: Option<&llama_cpp_2::model::ChatTemplateResult>,
) -> Option<Vec<(String, Value)>> {
    // Try the oaicompat parser first
    if let Some(tr) = template_result
        && let Ok(parsed_json) = tr.parse_response_oaicompat(output, false)
        && let Ok(value) = serde_json::from_str::<Value>(&parsed_json)
        && let Some(obj) = value.as_object()
        && let Some(tool_calls) = obj.get("tool_calls").and_then(Value::as_array)
    {
        let mut result = Vec::new();
        for tc in tool_calls {
            if let Some(function) = tc.get("function").and_then(Value::as_object) {
                let name = function.get("name").and_then(Value::as_str)?.to_string();
                let arguments = match function.get("arguments") {
                    Some(Value::String(s)) => {
                        serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                    }
                    Some(other) => other.clone(),
                    None => Value::Null,
                };
                result.push((name, arguments));
            }
        }
        if !result.is_empty() {
            return Some(result);
        }
    }

    // Fallback: parse XML tool call format used by some models (e.g. Qwen)
    // Format: <tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...
    parse_xml_tool_calls(output)
}

/// Parse XML-style tool calls emitted by some models (e.g. Qwen).
///
/// Example format:
/// ```text
/// <tool_call>
/// <function=write_file>
/// <parameter=path>
/// output.txt
/// </parameter>
/// <parameter=content>
/// Hello from LLM
/// </parameter>
/// </function>
/// </tool_call>
/// ```
fn parse_xml_tool_calls(output: &str) -> Option<Vec<(String, Value)>> {
    let mut results = Vec::new();

    for block in output.split("<tool_call>").skip(1) {
        let block = block.split("</tool_call>").next().unwrap_or(block);

        // Extract function name: <function=NAME>
        let func_start = block.find("<function=")?;
        let after_eq = &block[func_start + "<function=".len()..];
        let func_name_end = after_eq.find('>')?;
        let func_name = after_eq[..func_name_end].trim().to_string();

        // Extract parameters: <parameter=KEY>\nVALUE\n</parameter>
        let mut args = serde_json::Map::new();
        let mut search_from = 0;
        while let Some(param_start) = block[search_from..].find("<parameter=") {
            let abs_start = search_from + param_start;
            let after_param_eq = &block[abs_start + "<parameter=".len()..];
            let Some(key_end) = after_param_eq.find('>') else {
                break;
            };
            let key = after_param_eq[..key_end].trim();

            let value_start = abs_start + "<parameter=".len() + key_end + 1;
            let Some(param_end) = block[value_start..].find("</parameter>") else {
                break;
            };
            let value = block[value_start..value_start + param_end].trim();

            args.insert(key.to_string(), Value::String(value.to_string()));
            search_from = value_start + param_end + "</parameter>".len();
        }

        if !func_name.is_empty() {
            results.push((func_name, Value::Object(args)));
        }
    }

    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

pub(crate) fn parse_completion_output(
    raw_text: &str,
    template_result: Option<&llama_cpp_2::model::ChatTemplateResult>,
    has_json_schema: bool,
) -> Result<OneOrMany<AssistantContent>, String> {
    if crate::llama_logs_enabled() {
        eprintln!("[rig-llama-cpp] raw output:\n{raw_text}");
    }

    // When the caller set an output schema, grammar-constrained generation produces
    // a valid JSON object — but chat templates often wrap it in role tokens
    // (e.g. `<|im_start|>assistant\n`) or markdown fences (```json ... ```).
    // Strip those before any other parsing so Rig's typed prompt can deserialize.
    if has_json_schema
        && let Some(json) = extract_structured_json(raw_text)
    {
        return Ok(OneOrMany::one(AssistantContent::text(json)));
    }

    if let Some(template_result) = template_result {
        match template_result.parse_response_oaicompat(raw_text, false) {
            Ok(parsed_json) => {
                if crate::llama_logs_enabled() {
                    eprintln!("[rig-llama-cpp] parsed response: {parsed_json}");
                }
                if let Ok(choice) = parse_oaicompat_message(&parsed_json, raw_text) {
                    return Ok(choice);
                }
            }
            Err(err) => {
                eprintln!("Failed to parse llama response as OpenAI-compatible content: {err} (will try XML fallback)");
            }
        }
    }

    // Fallback: try XML tool call format before returning raw text
    if let Some(xml_tool_calls) = parse_xml_tool_calls(raw_text) {
        let mut content: Vec<AssistantContent> = Vec::new();
        for (i, (name, arguments)) in xml_tool_calls.into_iter().enumerate() {
            content.push(AssistantContent::ToolCall(ToolCall::new(
                format!("xml-tool-call-{i}"),
                ToolFunction::new(name, arguments),
            )));
        }
        if let Ok(result) = OneOrMany::many(content) {
            return Ok(result);
        }
    }

    Ok(OneOrMany::one(AssistantContent::text(raw_text.to_string())))
}

/// Extract a JSON object from grammar-constrained output that may be wrapped in
/// markdown code fences, ChatML role tokens, or extra prose.
///
/// Scans for the first `{` and returns the substring up to the matching `}`,
/// tracking brace depth and JSON string escaping so that braces inside strings
/// don't confuse the balance. Returns `None` if no balanced object is found.
pub(crate) fn extract_structured_json(raw_text: &str) -> Option<String> {
    let bytes = raw_text.as_bytes();
    let start = bytes.iter().position(|&b| b == b'{')?;

    let mut depth: usize = 0;
    let mut in_string = false;
    let mut escaped = false;

    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }

        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(raw_text[start..=i].to_string());
                }
            }
            _ => {}
        }
    }

    None
}

fn parse_oaicompat_message(
    parsed_json: &str,
    raw_text: &str,
) -> Result<OneOrMany<AssistantContent>, String> {
    let value: Value = serde_json::from_str(parsed_json)
        .map_err(|e| format!("Parsed response JSON deserialization failed: {e}"))?;
    let object = value
        .as_object()
        .ok_or_else(|| "Parsed response is not a JSON object".to_string())?;

    let mut content = Vec::new();

    if let Some(reasoning) = object
        .get("reasoning_content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        content.push(AssistantContent::Reasoning(Reasoning::new(reasoning)));
    }

    let text = extract_text_content(object.get("content"));
    if let Some(text) = text.filter(|text| !text.is_empty()) {
        content.push(AssistantContent::text(text));
    }

    if let Some(tool_calls) = object.get("tool_calls").and_then(Value::as_array) {
        for tool_call in tool_calls {
            content.push(AssistantContent::ToolCall(parse_tool_call(tool_call)?));
        }
    }

    if content.is_empty() {
        content.push(AssistantContent::text(raw_text.to_string()));
    }

    OneOrMany::many(content).map_err(|_| "Parsed response produced no content".to_string())
}

fn extract_text_content(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .or_else(|| part.get("refusal").and_then(Value::as_str))
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(text)
        }
        _ => None,
    }
}

fn parse_tool_call(value: &Value) -> Result<ToolCall, String> {
    let id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| "Tool call is missing id".to_string())?
        .to_string();
    let function = value
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| "Tool call is missing function".to_string())?;
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "Tool call function is missing name".to_string())?
        .to_string();
    let arguments = match function.get("arguments") {
        Some(Value::String(arguments)) => {
            serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.clone()))
        }
        Some(other) => other.clone(),
        None => Value::Null,
    };

    Ok(ToolCall::new(id, ToolFunction::new(name, arguments)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_structured_json_plain_object() {
        let out = extract_structured_json(r#"{"name":"Ada","age":36}"#).unwrap();
        assert_eq!(out, r#"{"name":"Ada","age":36}"#);
    }

    #[test]
    fn extract_structured_json_strips_markdown_fence() {
        let raw = "```json\n{\n  \"name\": \"Ada\",\n  \"age\": 36\n}\n```";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, "{\n  \"name\": \"Ada\",\n  \"age\": 36\n}");
    }

    #[test]
    fn extract_structured_json_strips_plain_fence() {
        let raw = "```\n{\"ok\": true}\n```";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"ok": true}"#);
    }

    #[test]
    fn extract_structured_json_strips_chatml_role_prefix() {
        let raw = "<|im_start|>assistant\n```json\n{\"value\": 1}\n```";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"value": 1}"#);
    }

    #[test]
    fn extract_structured_json_strips_leading_prose() {
        let raw = "Sure, here is the answer: {\"answer\": 42}";
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"answer": 42}"#);
    }

    #[test]
    fn extract_structured_json_handles_nested_objects() {
        let raw = r#"```json
{"person": {"name": "Ada", "skills": {"lang": "rust"}}, "age": 36}
```"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(
            out,
            r#"{"person": {"name": "Ada", "skills": {"lang": "rust"}}, "age": 36}"#
        );
    }

    #[test]
    fn extract_structured_json_ignores_braces_inside_strings() {
        let raw = r#"{"text": "an { inside } string", "ok": true}"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, raw);
    }

    #[test]
    fn extract_structured_json_handles_escaped_quotes_in_strings() {
        let raw = r#"{"text": "she said \"hi\"", "brace": "}"}"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, raw);
    }

    #[test]
    fn extract_structured_json_stops_at_first_balanced_object() {
        let raw = r#"{"first": 1} and then {"second": 2}"#;
        let out = extract_structured_json(raw).unwrap();
        assert_eq!(out, r#"{"first": 1}"#);
    }

    #[test]
    fn extract_structured_json_returns_none_when_unbalanced() {
        assert!(extract_structured_json(r#"{"broken": "#).is_none());
    }

    #[test]
    fn extract_structured_json_returns_none_when_no_object() {
        assert!(extract_structured_json("just plain text, no json").is_none());
        assert!(extract_structured_json("").is_none());
    }

    #[test]
    fn extract_structured_json_handles_real_qwen_output() {
        // Shape observed in practice: ChatML role token, markdown fence, indented body.
        let raw = "<|im_start|>assistant\n```json\n{\n  \"age\": 36,\n  \"name\": \"Ada\",\n  \"occupation\": \"Software Engineer\"\n}\n```";
        let out = extract_structured_json(raw).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["name"], "Ada");
        assert_eq!(parsed["age"], 36);
        assert_eq!(parsed["occupation"], "Software Engineer");
    }
}
