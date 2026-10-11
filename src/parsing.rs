use serde_json::Value;

use crate::types::OutputPart;

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

/// The tool schemas offered with the current request, kept around so the
/// parameter form can be typed rather than guessed.
///
/// That form carries no types at all — `<parameter=fields>{"Vendor":"X"}` is
/// just text — so a parameter declared as an object has to be parsed back into
/// one, while a parameter declared as a string must be left exactly as written
/// even when its text happens to read as JSON. Only the schema can tell those
/// apart, so it travels with the parse.
pub(crate) struct ToolSchemas(Value);

impl ToolSchemas {
    /// Parse the request's `tools_json`. `None` when there are no tools or the
    /// JSON is unusable — parsing then falls back to a conservative guess.
    pub(crate) fn parse(tools_json: Option<&str>) -> Option<Self> {
        let tools: Value = serde_json::from_str(tools_json?).ok()?;
        tools.is_array().then_some(Self(tools))
    }

    /// The declared JSON-Schema `type` of one parameter, if the request
    /// described it. A union type (`["string", "null"]`) reports its first
    /// non-`null` member, which is what the value has to deserialize into.
    fn parameter_type(&self, function: &str, key: &str) -> Option<&str> {
        let declared = self
            .0
            .as_array()?
            .iter()
            .map(|tool| tool.get("function").unwrap_or(tool))
            .find(|f| f.get("name").and_then(Value::as_str) == Some(function))?
            .get("parameters")?
            .get("properties")?
            .get(key)?
            .get("type")?;
        match declared {
            Value::String(name) => Some(name.as_str()),
            Value::Array(names) => names
                .iter()
                .filter_map(Value::as_str)
                .find(|name| *name != "null"),
            _ => None,
        }
    }
}

/// Turn one parameter-form value into the JSON the tool's deserializer expects.
///
/// `declared` is the parameter's schema type when the request supplied one.
/// With it the answer is exact: a `string` parameter keeps its text verbatim,
/// anything else is parsed. Without it — an unknown tool, or a request that
/// offered no schema — only objects and arrays are taken, so free text that
/// happens to read as `2`, `true` or `null` is left alone.
fn coerce_parameter(raw: &str, declared: Option<&str>) -> Value {
    if declared == Some("string") {
        return Value::String(raw.to_string());
    }
    match serde_json::from_str::<Value>(raw) {
        Ok(parsed) if declared.is_some() => parsed,
        Ok(parsed @ (Value::Object(_) | Value::Array(_))) => parsed,
        _ => Value::String(raw.to_string()),
    }
}

pub(crate) fn parse_completion_output(
    raw_text: &str,
    has_json_schema: bool,
    schemas: Option<&ToolSchemas>,
) -> Vec<OutputPart> {
    log::debug!("raw output:\n{raw_text}");

    // When the caller set an output schema, grammar-constrained generation produces
    // a valid JSON object — but chat templates often wrap it in role tokens
    // (e.g. `<|im_start|>assistant\n`) or markdown fences (```json ... ```).
    // Strip those before any other parsing so Rig's typed prompt can deserialize.
    if has_json_schema && let Some(json) = extract_structured_json(raw_text) {
        return vec![OutputPart::Text(json)];
    }

    // Try to surface tool calls. The OpenAI-compatible
    // `parse_response_oaicompat` path from llama-cpp-2 0.1.146 was removed in
    // 0.1.147, so we rely on our own parsers here (XML `<tool_call>` blocks,
    // bare / markdown-fenced JSON).
    if let Some(tool_calls) = parse_tool_calls(raw_text, schemas)
        && !tool_calls.is_empty()
    {
        return tool_calls
            .into_iter()
            .map(|(name, arguments)| OutputPart::ToolCall { name, arguments })
            .collect();
    }

    // Try to surface reasoning wrapped in <think>...</think> before giving up
    // on structured content.
    if let Some((reasoning, text)) = split_thinking(raw_text) {
        let mut content = Vec::new();
        if !reasoning.is_empty() {
            content.push(OutputPart::Reasoning(reasoning));
        }
        content.push(OutputPart::Text(text));
        return content;
    }

    vec![OutputPart::Text(raw_text.to_string())]
}

/// Unified tool-call parser. Tries, in order:
///
/// 1. `<tool_call>` blocks (Qwen-style XML parameter form or JSON form).
/// 2. Gemma-4's native `<|tool_call>call:NAME{...}<tool_call|>` DSL.
/// 3. Bare / markdown-fenced JSON objects shaped like
///    `{"name": ..., "arguments": ...}`.
///
/// Returns the first format that yields at least one tool call, so a response
/// that mixes prose and a single JSON tool call still parses.
pub(crate) fn parse_tool_calls(
    output: &str,
    schemas: Option<&ToolSchemas>,
) -> Option<Vec<(String, Value)>> {
    if let Some(xml) = parse_xml_tool_calls(output, schemas) {
        return Some(xml);
    }
    if let Some(gemma) = parse_gemma_tool_calls(output) {
        return Some(gemma);
    }
    // Bare / markdown-fenced JSON. Models that ignore the `<tool_call>`
    // directive often emit a fenced ```json block instead.
    if let Some(json) = extract_structured_json(output)
        && let Ok(value) = serde_json::from_str::<Value>(&json)
        && let Some(tc) = json_value_to_tool_call(&value)
    {
        return Some(vec![tc]);
    }
    None
}

/// Parse XML-style tool calls emitted by some models (e.g. Qwen).
///
/// Two shapes are recognised:
///
/// 1. Parameter form:
/// ```text
/// <tool_call>
/// <function=write_file>
/// <parameter=path>
/// output.txt
/// </parameter>
/// </function>
/// </tool_call>
/// ```
///
/// 2. JSON form (emitted by the system-prompt directive in `prompt.rs`):
/// ```text
/// <tool_call>
/// {"name": "write_file", "arguments": {"path": "output.txt"}}
/// </tool_call>
/// ```
pub(crate) fn parse_xml_tool_calls(
    output: &str,
    schemas: Option<&ToolSchemas>,
) -> Option<Vec<(String, Value)>> {
    let mut results = Vec::new();

    for block in output.split("<tool_call>").skip(1) {
        let block = block.split("</tool_call>").next().unwrap_or(block);
        let trimmed = block.trim();

        // JSON form first.
        if trimmed.starts_with('{')
            && let Ok(value) = serde_json::from_str::<Value>(trimmed)
            && let Some(tc) = json_value_to_tool_call(&value)
        {
            results.push(tc);
            continue;
        }

        // Parameter form fallback.
        if let Some(tc) = parse_parameter_form(block, schemas) {
            results.push(tc);
        }
    }

    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

/// Gemma-4 opens and closes strings with this same token, quote-style.
const GEMMA_STRING_DELIM: &str = "<|\"|>";

/// Tool-call markers, longest-first so `<|tool_call>` is never mistaken for
/// `<tool_call>`. Used to locate prose preceding the first call.
pub(crate) const TOOL_CALL_MARKERS: &[&str] = &["<|tool_call>", "<tool_call>"];

/// Parse Gemma-4's native tool-call blocks.
///
/// Gemma reverts to this format — rather than the portable `<tool_call>` JSON
/// directive injected into the system prompt — whenever it is prompted in its
/// own turn format, because that is what it was trained to emit:
///
/// ```text
/// <|tool_call>call:write_file{path:<|"|>a.txt<|"|>,retries:2}<tool_call|>
/// ```
///
/// Arguments are a bespoke DSL, not JSON: strings are wrapped in
/// `<|"|>` on both sides, `null` / `true` / `false` and bare numbers are
/// literals, and objects/arrays nest with `{k:v,...}` / `[a,b]`. Top-level
/// argument keys are unquoted; nested keys are string-wrapped.
pub(crate) fn parse_gemma_tool_calls(output: &str) -> Option<Vec<(String, Value)>> {
    let mut results = Vec::new();

    for block in output.split("<|tool_call>").skip(1) {
        let block = block.split("<tool_call|>").next().unwrap_or(block).trim();
        // The template writes `call:` before the function name.
        let block = block.strip_prefix("call:").unwrap_or(block);

        let Some(brace) = block.find('{') else {
            continue;
        };
        let name = block[..brace].trim();
        if name.is_empty() {
            continue;
        }

        let mut at = brace;
        let arguments = parse_gemma_value(block, &mut at).unwrap_or(Value::Null);
        results.push((name.to_string(), arguments));
    }

    if results.is_empty() {
        None
    } else {
        Some(results)
    }
}

fn skip_gemma_ws(s: &str, at: &mut usize) {
    while let Some(c) = s[*at..].chars().next() {
        if c.is_whitespace() {
            *at += c.len_utf8();
        } else {
            break;
        }
    }
}

/// Read one value of Gemma's argument DSL starting at `at`, advancing it past
/// what was consumed.
fn parse_gemma_value(s: &str, at: &mut usize) -> Option<Value> {
    skip_gemma_ws(s, at);
    let rest = s.get(*at..)?;

    if let Some(body) = rest.strip_prefix(GEMMA_STRING_DELIM) {
        let end = body.find(GEMMA_STRING_DELIM)?;
        *at += GEMMA_STRING_DELIM.len() + end + GEMMA_STRING_DELIM.len();
        return Some(Value::String(body[..end].to_string()));
    }

    if rest.starts_with('{') {
        *at += 1;
        let mut map = serde_json::Map::new();
        loop {
            skip_gemma_ws(s, at);
            let rest = s.get(*at..)?;
            if let Some(stripped) = rest.strip_prefix('}') {
                let _ = stripped;
                *at += 1;
                break;
            }
            if rest.is_empty() {
                return None;
            }
            let key = parse_gemma_key(s, at)?;
            skip_gemma_ws(s, at);
            if !s.get(*at..)?.starts_with(':') {
                return None;
            }
            *at += 1;
            let value = parse_gemma_value(s, at)?;
            map.insert(key, value);
            skip_gemma_ws(s, at);
            if s.get(*at..)?.starts_with(',') {
                *at += 1;
            }
        }
        return Some(Value::Object(map));
    }

    if rest.starts_with('[') {
        *at += 1;
        let mut items = Vec::new();
        loop {
            skip_gemma_ws(s, at);
            let rest = s.get(*at..)?;
            if rest.starts_with(']') {
                *at += 1;
                break;
            }
            if rest.is_empty() {
                return None;
            }
            items.push(parse_gemma_value(s, at)?);
            skip_gemma_ws(s, at);
            if s.get(*at..)?.starts_with(',') {
                *at += 1;
            }
        }
        return Some(Value::Array(items));
    }

    // Bare scalar: runs until the next structural character.
    let end = rest.find([',', '}', ']']).unwrap_or(rest.len());
    let token = rest[..end].trim();
    *at += end;
    Some(match token {
        "null" => Value::Null,
        "true" => Value::Bool(true),
        "false" => Value::Bool(false),
        _ => token
            .parse::<i64>()
            .map(Value::from)
            .or_else(|_| token.parse::<f64>().map(Value::from))
            .unwrap_or_else(|_| Value::String(token.to_string())),
    })
}

/// Object keys are either string-wrapped (nested objects) or bare (top-level
/// argument names, which the template renders with `escape_keys=False`).
fn parse_gemma_key(s: &str, at: &mut usize) -> Option<String> {
    skip_gemma_ws(s, at);
    let rest = s.get(*at..)?;

    if let Some(body) = rest.strip_prefix(GEMMA_STRING_DELIM) {
        let end = body.find(GEMMA_STRING_DELIM)?;
        *at += GEMMA_STRING_DELIM.len() + end + GEMMA_STRING_DELIM.len();
        return Some(body[..end].to_string());
    }

    let end = rest.find(':')?;
    *at += end;
    let key = rest[..end].trim();
    if key.is_empty() {
        None
    } else {
        Some(key.to_string())
    }
}

/// Extract a single `(name, arguments)` tool call from a JSON value that is
/// either `{"name": ..., "arguments": ...}` or the OpenAI-shaped
/// `{"function": {"name": ..., "arguments": ...}}`.
fn json_value_to_tool_call(value: &Value) -> Option<(String, Value)> {
    let name = value.get("name").and_then(Value::as_str).or_else(|| {
        value
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(Value::as_str)
    })?;
    let arguments = match value.get("arguments") {
        Some(Value::String(s)) => {
            serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
        }
        Some(other) => other.clone(),
        None => value
            .get("function")
            .and_then(|f| f.get("arguments"))
            .map(|a| match a {
                Value::String(s) => {
                    serde_json::from_str(s).unwrap_or_else(|_| Value::String(s.clone()))
                }
                other => other.clone(),
            })
            .unwrap_or(Value::Null),
    };
    Some((name.to_string(), arguments))
}

/// Parse the Qwen parameter form (`<function=NAME>` + `<parameter=KEY>` blocks).
fn parse_parameter_form(block: &str, schemas: Option<&ToolSchemas>) -> Option<(String, Value)> {
    let func_start = block.find("<function=")?;
    let after_eq = &block[func_start + "<function=".len()..];
    let func_name_end = after_eq.find('>')?;
    let func_name = after_eq[..func_name_end].trim().to_string();

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

        let declared = schemas.and_then(|s| s.parameter_type(&func_name, key));
        args.insert(key.to_string(), coerce_parameter(value, declared));
        search_from = value_start + param_end + "</parameter>".len();
    }

    if func_name.is_empty() {
        None
    } else {
        Some((func_name, Value::Object(args)))
    }
}

/// Reasoning delimiters, as `(open, close)` pairs, for the model families this
/// crate targets. Models disagree on the markup, so we recognise each shape
/// rather than assuming the `<think>` convention is universal.
const THINKING_MARKERS: &[(&str, &str)] = &[
    // Qwen, DeepSeek-R1, and most reasoning models.
    ("<think>", "</think>"),
    // Gemma-4's thought channel.
    ("<|channel>thought", "<channel|>"),
];

/// Split reasoning from the visible response. Returns `(reasoning, text)` when
/// a thinking block is present, using whichever known marker pair appears
/// first. An unterminated block — the token cap hit mid-thought — is treated as
/// reasoning all the way to the end, leaving no visible text.
fn split_thinking(output: &str) -> Option<(String, String)> {
    let (open, close, start) = THINKING_MARKERS
        .iter()
        .filter_map(|&(open, close)| output.find(open).map(|at| (open, close, at)))
        .min_by_key(|&(_, _, at)| at)?;

    let body = start + open.len();
    let (reasoning, tail) = match output[body..].find(close) {
        Some(rel) => (
            &output[body..body + rel],
            &output[body + rel + close.len()..],
        ),
        None => (&output[body..], ""),
    };

    let mut text = String::with_capacity(start + tail.len());
    text.push_str(&output[..start]);
    text.push_str(tail);
    // Defensive: drop any stray closer left by a second block.
    let text = text.replace(close, "");
    Some((reasoning.trim().to_string(), text.trim().to_string()))
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

    #[test]
    fn parse_xml_tool_calls_json_form() {
        let raw = "<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"UTC\"}}\n</tool_call>";
        let out = parse_xml_tool_calls(raw, None).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "get_time");
        assert_eq!(out[0].1["timezone"], "UTC");
    }

    #[test]
    fn parse_xml_tool_calls_parameter_form() {
        let raw = "<tool_call>\n<function=write_file>\n<parameter=path>\noutput.txt\n</parameter>\n<parameter=content>\nHello\n</parameter>\n</function>\n</tool_call>";
        let out = parse_xml_tool_calls(raw, None).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "write_file");
        assert_eq!(out[0].1["path"], "output.txt");
        assert_eq!(out[0].1["content"], "Hello");
    }

    /// The schema for a tool taking an object, a string, and a number — the
    /// three cases the parameter form cannot distinguish on its own.
    fn update_entity_schemas() -> ToolSchemas {
        ToolSchemas::parse(Some(
            &serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "update_entity",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "fields": {"type": "object"},
                            "id": {"type": "string"},
                            "revision": {"type": "integer"},
                            "tags": {"type": "array"},
                        },
                    },
                },
            }])
            .to_string(),
        ))
        .unwrap()
    }

    #[test]
    fn parameter_form_types_its_values_from_the_tool_schema() {
        // Qwen3.6 emits this shape, and every value in it is bare text. Without
        // the schema `fields` deserializes as a string and the tool rejects it
        // with "invalid type: string ..., expected a map".
        let raw = "<tool_call>\n<function=update_entity>\n\
                   <parameter=id>\nABB1-NTP1\n</parameter>\n\
                   <parameter=fields>\n{\"Vendor\": \"Meinberg\"}\n</parameter>\n\
                   <parameter=revision>\n7\n</parameter>\n\
                   <parameter=tags>\n[\"ntp\"]\n</parameter>\n\
                   </function>\n</tool_call>";
        let out = parse_xml_tool_calls(raw, Some(&update_entity_schemas())).unwrap();
        assert_eq!(out[0].0, "update_entity");
        assert_eq!(
            out[0].1["fields"],
            serde_json::json!({"Vendor": "Meinberg"})
        );
        assert_eq!(out[0].1["id"], serde_json::json!("ABB1-NTP1"));
        assert_eq!(out[0].1["revision"], serde_json::json!(7));
        assert_eq!(out[0].1["tags"], serde_json::json!(["ntp"]));
    }

    #[test]
    fn a_string_parameter_keeps_json_looking_text_verbatim() {
        // The other half of the same problem: a declared string must survive
        // even when its content reads as JSON, or a tool that writes a script
        // or a document gets a parsed object instead of the text.
        let raw = "<tool_call>\n<function=update_entity>\n\
                   <parameter=id>\n{\"not\": \"an object\"}\n</parameter>\n\
                   </function>\n</tool_call>";
        let out = parse_xml_tool_calls(raw, Some(&update_entity_schemas())).unwrap();
        assert_eq!(
            out[0].1["id"],
            serde_json::json!("{\"not\": \"an object\"}")
        );
    }

    #[test]
    fn without_a_schema_only_objects_and_arrays_are_parsed() {
        // No schema to lean on, so structure is recovered but a bare `7` stays
        // text rather than being guessed into a number.
        let raw = "<tool_call>\n<function=unknown>\n\
                   <parameter=fields>\n{\"Vendor\": \"Meinberg\"}\n</parameter>\n\
                   <parameter=revision>\n7\n</parameter>\n\
                   </function>\n</tool_call>";
        let out = parse_xml_tool_calls(raw, None).unwrap();
        assert_eq!(
            out[0].1["fields"],
            serde_json::json!({"Vendor": "Meinberg"})
        );
        assert_eq!(out[0].1["revision"], serde_json::json!("7"));
    }

    #[test]
    fn a_nullable_parameter_uses_its_non_null_type() {
        let schemas = ToolSchemas::parse(Some(
            &serde_json::json!([{
                "type": "function",
                "function": {
                    "name": "t",
                    "parameters": {"properties": {"note": {"type": ["string", "null"]}}},
                },
            }])
            .to_string(),
        ))
        .unwrap();
        let raw = "<tool_call>\n<function=t>\n<parameter=note>\n{\"a\": 1}\n</parameter>\n\
                   </function>\n</tool_call>";
        let out = parse_xml_tool_calls(raw, Some(&schemas)).unwrap();
        assert_eq!(out[0].1["note"], serde_json::json!("{\"a\": 1}"));
    }

    #[test]
    fn parse_xml_tool_calls_multiple_blocks() {
        let raw = "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>\n<tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>";
        let out = parse_xml_tool_calls(raw, None).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "a");
        assert_eq!(out[1].0, "b");
    }

    #[test]
    fn parse_gemma_tool_calls_no_arguments() {
        // The shape Gemma-4 actually emitted in the e2e tool roundtrip.
        let raw =
            "<|channel>thought\nI should call it.<channel|><|tool_call>call:get_time{}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, "get_time");
        assert_eq!(out[0].1, serde_json::json!({}));
    }

    #[test]
    fn parse_gemma_tool_calls_scalar_arguments() {
        let raw = "<|tool_call>call:write_file{path:<|\"|>a.txt<|\"|>,retries:2,force:true,note:null}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out[0].0, "write_file");
        assert_eq!(out[0].1["path"], "a.txt");
        assert_eq!(out[0].1["retries"], 2);
        assert_eq!(out[0].1["force"], true);
        assert_eq!(out[0].1["note"], Value::Null);
    }

    #[test]
    fn parse_gemma_tool_calls_nested_and_sequence_arguments() {
        // Nested keys are string-wrapped; sequences use [a,b].
        let raw = "<|tool_call>call:cfg{opts:{<|\"|>depth<|\"|>:3},tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out[0].1["opts"]["depth"], 3);
        assert_eq!(out[0].1["tags"], serde_json::json!(["a", "b"]));
    }

    #[test]
    fn parse_gemma_tool_calls_multiple_blocks() {
        let raw = "<|tool_call>call:a{}<tool_call|><|tool_call>call:b{x:1}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].0, "a");
        assert_eq!(out[1].0, "b");
        assert_eq!(out[1].1["x"], 1);
    }

    #[test]
    fn parse_gemma_tool_calls_preserves_braces_inside_strings() {
        let raw = "<|tool_call>call:echo{msg:<|\"|>a{b},c<|\"|>}<tool_call|>";
        let out = parse_gemma_tool_calls(raw).unwrap();
        assert_eq!(out[0].1["msg"], "a{b},c");
    }

    #[test]
    fn parse_gemma_tool_calls_returns_none_without_blocks() {
        assert!(parse_gemma_tool_calls("just prose").is_none());
        // A Qwen-style block must not be picked up by the Gemma parser.
        assert!(parse_gemma_tool_calls("<tool_call>{\"name\":\"a\"}</tool_call>").is_none());
    }

    #[test]
    fn parse_tool_calls_dispatches_to_the_gemma_dialect() {
        let raw = "<|tool_call>call:get_time{}<tool_call|>";
        let out = parse_tool_calls(raw, None).unwrap();
        assert_eq!(out[0].0, "get_time");
    }

    #[test]
    fn parse_completion_output_surfaces_gemma_tool_call() {
        let raw = "<|channel>thought\nreasoning<channel|><|tool_call>call:get_time{}<tool_call|>";
        let content = parse_completion_output(raw, false, None);
        let name = content.iter().find_map(|c| match c {
            OutputPart::ToolCall { name, .. } => Some(name.as_str()),
            _ => None,
        });
        assert_eq!(name.expect("expected a tool call"), "get_time");
    }

    #[test]
    fn split_thinking_extracts_reasoning() {
        let raw = "<think>let me consider</think>The answer is 42.";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "let me consider");
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    fn split_thinking_extracts_gemma_thought_channel() {
        let raw = "<|channel>thought\nweigh the options\n<channel|>The answer is 42.";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "weigh the options");
        assert_eq!(text, "The answer is 42.");
    }

    #[test]
    fn split_thinking_handles_unterminated_block() {
        // Token cap hit mid-thought: everything is reasoning, no visible text.
        let raw = "<think>still working through it";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "still working through it");
        assert_eq!(text, "");
    }

    #[test]
    fn split_thinking_keeps_prose_before_the_block() {
        let raw = "Sure.<think>hmm</think>Done.";
        let (reasoning, text) = split_thinking(raw).unwrap();
        assert_eq!(reasoning, "hmm");
        assert_eq!(text, "Sure.Done.");
    }

    #[test]
    fn split_thinking_returns_none_without_markers() {
        assert!(split_thinking("just a plain answer").is_none());
    }

    #[test]
    fn parse_completion_output_surfaces_gemma_reasoning() {
        let raw = "<|channel>thought\nreason about it\n<channel|>Because of Rayleigh scattering.";
        let content = parse_completion_output(raw, false, None);
        let has_reasoning = content
            .iter()
            .any(|c| matches!(c, OutputPart::Reasoning(_)));
        assert!(
            has_reasoning,
            "expected reasoning content, got: {content:?}"
        );
    }
}
