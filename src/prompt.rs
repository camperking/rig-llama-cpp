use serde_json::{Map, Value, json};

use crate::types::{PreparedRequest, PromptBuildResult};

/// Render a [`PreparedRequest`] into a prompt string.
///
/// Four paths, in order:
///
/// 0. **The model's own template, fed native tool structures**
///    ([`crate::jinja`]) — taken only for a turn that involves tools, because
///    it is the only path that can represent one faithfully.
/// 1. **llama.cpp's applier** — the fast path for every tool-free turn whose
///    template it recognises. Unchanged from before path 0 existed.
/// 2. **minijinja** — for models whose bespoke jinja template llama.cpp cannot
///    apply. Also forwards `enable_thinking`, which the C API cannot pass.
/// 3. **ChatML** — last resort, for a model with no usable template at all.
///
/// `llama-cpp-2` 0.1.147 removed the `apply_chat_template_oaicompat` /
/// `OpenAIChatTemplateParams` path that previously let llama.cpp's jinja engine
/// ingest `tools` / `tool_choice` / `json_schema` directly. What remains takes
/// only `(role, content)` pairs — and `tool_calls` is a *sibling* of `content`,
/// not part of it. Flattening a message to a pair therefore erases the
/// assistant's own tool calls, leaving the model an empty assistant turn
/// followed by a tool result it has no record of asking for. Models handle that
/// by size: a small one stops, a large one guesses. Path 0 restores what the
/// oaicompat path used to do — hand the template the structured messages and
/// the `tools` array, and let it render its own protocol.
///
/// Paths 1-3 stay lossy by construction, so [`flatten_messages`] folds tool
/// calls back into the message text as `<tool_call>` blocks matching
/// [`build_tool_directive`]. Degraded against a model with its own dialect, but
/// never silent.
///
/// JSON-schema constraints are applied as a sampler (see `src/sampling.rs`)
/// rather than baked into the prompt.
pub(crate) fn build_prompt(
    model: &llama_cpp_2::model::LlamaModel,
    request: &PreparedRequest,
) -> Result<PromptBuildResult, String> {
    use llama_cpp_2::model::LlamaChatMessage;

    let raw_messages: Vec<Value> = serde_json::from_str(&request.messages_json)
        .map_err(|e| format!("Message deserialization failed: {e}"))?;

    let template = model.chat_template(None).ok();

    // Path 0: the model's own template, with tool calls intact.
    if let Some(tmpl) = template.as_ref()
        && involves_tools(request, &raw_messages)
    {
        match render_with_native_tools(tmpl, &raw_messages, request) {
            Ok(prompt) => {
                log::debug!("messages_json: {}", request.messages_json);
                log::debug!("rendered prompt (minijinja, native tools):\n{prompt}");
                return Ok(PromptBuildResult { prompt });
            }
            Err(e) => log::warn!(
                "the model's own template could not render this tool-calling turn ({e}); \
                 falling back to the portable <tool_call> directive"
            ),
        }
    }

    let parsed_messages = flatten_messages(&raw_messages);

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

    // Path 1: llama.cpp's built-in applier.
    let applied = template.as_ref().and_then(|tmpl| {
        let chat_msgs: Vec<LlamaChatMessage> = rendered_messages
            .iter()
            .map(|(role, content)| LlamaChatMessage::new(role.clone(), content.clone()))
            .collect::<Result<_, _>>()
            .ok()?;
        match model.apply_chat_template(tmpl, &chat_msgs, true) {
            Ok(prompt) => Some(prompt),
            Err(e) => {
                // `apply_chat_template` is llama.cpp's *non-jinja* applier: it
                // recognises a fixed set of known template shapes and returns
                // -1 for anything else. Models with a bespoke jinja template
                // (Gemma-4's `<|turn>role` format, for one) land here and fall
                // through to minijinja below.
                log::debug!("llama.cpp could not apply the chat template ({e}); trying minijinja");
                None
            }
        }
    });

    if let Some(prompt) = applied {
        log::debug!("messages_json: {}", request.messages_json);
        log::debug!("rendered prompt:\n{prompt}");
        return Ok(PromptBuildResult { prompt });
    }

    // Path 2: render the model's real jinja template ourselves, using the same
    // flattened messages as path 1 so only the turn format and the template
    // variables change.
    if let Some(tmpl) = template.as_ref() {
        match render_with_jinja(tmpl, &rendered_messages, request) {
            Ok(prompt) => {
                log::debug!("rendered prompt (minijinja):\n{prompt}");
                return Ok(PromptBuildResult { prompt });
            }
            Err(e) => log::warn!(
                "model ships a chat template neither llama.cpp nor minijinja could render \
                 ({e}); falling back to ChatML. Output quality will suffer if the model does \
                 not use a ChatML-compatible format."
            ),
        }
    } else {
        log::warn!("model ships no chat template; falling back to ChatML.");
    }

    // Fallback to ChatML.
    let mut prompt = String::new();
    for (role, content) in &rendered_messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    Ok(PromptBuildResult { prompt })
}

/// Whether this turn needs the tool-faithful path.
///
/// True when the request offers tools *or* the history already contains tool
/// traffic — the follow-up turn after a tool result carries no `tools` of its
/// own, yet still has an assistant `tool_calls` message that must survive.
fn involves_tools(request: &PreparedRequest, messages: &[Value]) -> bool {
    request.tools_json.is_some()
        || messages.iter().any(|msg| {
            msg.get("tool_calls").is_some_and(|calls| !calls.is_null())
                || msg.get("role").and_then(Value::as_str) == Some("tool")
        })
}

/// Render the model's own template with `tool_calls`, tool results and the
/// `tools` array left structured, so the template emits its native protocol.
fn render_with_native_tools(
    template: &llama_cpp_2::model::LlamaChatTemplate,
    raw_messages: &[Value],
    request: &PreparedRequest,
) -> Result<String, String> {
    let template = template
        .to_str()
        .map_err(|e| format!("chat template is not valid UTF-8: {e}"))?;

    let tools: Option<Value> = request
        .tools_json
        .as_deref()
        .map(|json| {
            serde_json::from_str(json).map_err(|e| format!("tools are not valid JSON: {e}"))
        })
        .transpose()?;

    let mut messages = structured_messages(raw_messages);
    if let Some(note) = tool_choice_note(request.tool_choice.as_deref()) {
        // Templates render `tools`, but none of them has a `tool_choice`
        // variable; the constraint has to travel as prose.
        merge_system_note(&mut messages, &note);
    }

    crate::jinja::render_chat_template(
        template,
        &messages,
        &crate::jinja::TemplateVars {
            tools: tools.as_ref(),
            enable_thinking: request.enable_thinking,
            bos_token: "",
        },
    )
    .map_err(|e| format!("{e:#}"))
}

/// Flatten each message's `content` to text while keeping the fields a chat
/// template reads: `tool_calls` on an assistant turn and `tool_call_id` on a
/// tool result.
///
/// Content is flattened even here, deliberately — see
/// [`crate::jinja::render_chat_template`] on why raw content parts break
/// multimodal marker/bitmap balance.
fn structured_messages(raw: &[Value]) -> Vec<Value> {
    let mut out = Vec::with_capacity(raw.len());
    for msg in raw {
        let Some(role) = msg.get("role").and_then(Value::as_str) else {
            continue;
        };
        let mut rendered = Map::new();
        rendered.insert("role".to_string(), json!(role));
        rendered.insert("content".to_string(), json!(message_content_as_text(msg)));
        if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
            rendered.insert(
                "tool_calls".to_string(),
                Value::Array(calls.iter().map(template_tool_call).collect()),
            );
        }
        if let Some(id) = msg.get("tool_call_id") {
            rendered.insert("tool_call_id".to_string(), id.clone());
        }
        out.push(Value::Object(rendered));
    }
    out
}

/// Re-shape one `tool_calls` entry for a chat template.
///
/// `src/request.rs` serializes `function.arguments` as a JSON-encoded *string*,
/// which is what the OpenAI wire format uses. Chat templates overwhelmingly
/// expect an object and render it with `| tojson`; handed a string, that filter
/// emits a quoted, escaped blob and the model learns to call tools wrongly.
/// Templates that do guard with `arguments is string` are equally happy with an
/// object, so parsing it back is the portable choice. Anything that fails to
/// parse is left exactly as it came.
fn template_tool_call(call: &Value) -> Value {
    let mut call = call.clone();
    if let Some(function) = call.get_mut("function")
        && let Some(arguments) = function.get_mut("arguments")
    {
        let parsed = arguments
            .as_str()
            .and_then(|text| serde_json::from_str::<Value>(text).ok());
        if let Some(parsed) = parsed {
            *arguments = parsed;
        }
    }
    call
}

/// The prose form of a non-`auto` tool choice, or `None` when the default
/// applies.
fn tool_choice_note(tool_choice: Option<&str>) -> Option<String> {
    let choice = tool_choice.filter(|choice| *choice != "auto")?;
    let how = match choice {
        "none" => "Do not call any tools for this turn.",
        "required" => "You must call at least one tool in your response.",
        other => other,
    };
    Some(format!("Tool choice: {how}"))
}

/// Append `note` to the first system message, creating one if there is none.
fn merge_system_note(messages: &mut Vec<Value>, note: &str) {
    let system = messages
        .iter()
        .position(|msg| msg.get("role").and_then(Value::as_str) == Some("system"));
    match system {
        Some(at) => {
            let existing = messages[at]
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("");
            let merged = if existing.trim().is_empty() {
                note.to_string()
            } else {
                format!("{existing}\n\n{note}")
            };
            messages[at]["content"] = json!(merged);
        }
        None => messages.insert(0, json!({ "role": "system", "content": note })),
    }
}

/// Flatten messages to the `(role, content)` pairs paths 1-3 accept.
///
/// Tool traffic has no representation in that shape, so it is rewritten into
/// text using the same portable protocol [`build_tool_directive`] asks the
/// model for: an assistant's calls become `<tool_call>` blocks appended to its
/// text, and a tool result becomes a user turn wrapped in `<tool_response>`.
/// The round trip is lossy — a model with its own dialect sees a protocol it
/// was not trained on — but it keeps the call in the transcript, which is what
/// lets the model make a second one.
fn flatten_messages(raw: &[Value]) -> Vec<(String, String)> {
    let mut out = Vec::with_capacity(raw.len());
    for msg in raw {
        let Some(role) = msg.get("role").and_then(Value::as_str) else {
            continue;
        };
        let text = message_content_as_text(msg);

        if role == "tool" {
            out.push((
                "user".to_string(),
                format!("<tool_response>\n{text}\n</tool_response>"),
            ));
            continue;
        }

        let calls = msg
            .get("tool_calls")
            .and_then(Value::as_array)
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(render_tool_call_block)
                    .collect::<Vec<_>>()
                    .join("\n")
            })
            .unwrap_or_default();

        let content = match (text.is_empty(), calls.is_empty()) {
            (_, true) => text,
            (true, false) => calls,
            (false, false) => format!("{text}\n{calls}"),
        };
        out.push((role.to_string(), content));
    }
    out
}

/// One tool call as the `<tool_call>` JSON block [`build_tool_directive`]
/// specifies and `crate::parsing::parse_xml_tool_calls` reads back.
fn render_tool_call_block(call: &Value) -> Option<String> {
    let function = call.get("function")?;
    let name = function.get("name").and_then(Value::as_str)?;
    // `arguments` is a JSON-encoded string here (see `template_tool_call`);
    // it is already in the form the block wants.
    let arguments = match function.get("arguments") {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some(format!(
        "<tool_call>\n{{\"name\": \"{name}\", \"arguments\": {arguments}}}\n</tool_call>"
    ))
}

/// Render the model's own jinja chat template, supplying the variables the
/// llama.cpp C API has no way to forward.
///
/// `messages` are the same flattened, tool-directive-injected pairs the other
/// two paths use — see [`crate::jinja::render_chat_template`] for why the raw
/// structured messages are handed over only on the tool path.
///
/// The BOS piece is deliberately passed as an empty string: templates emit
/// `bos_token` explicitly, but tokenization already prepends BOS
/// (`AddBos::Always` in `worker.rs`), and emitting it here too would duplicate
/// it at position 0.
fn render_with_jinja(
    template: &llama_cpp_2::model::LlamaChatTemplate,
    messages: &[(String, String)],
    request: &PreparedRequest,
) -> Result<String, String> {
    let template = template
        .to_str()
        .map_err(|e| format!("chat template is not valid UTF-8: {e}"))?;

    let messages: Vec<Value> = messages
        .iter()
        .map(|(role, content)| json!({ "role": role, "content": content }))
        .collect();

    crate::jinja::render_chat_template(
        template,
        &messages,
        &crate::jinja::TemplateVars {
            tools: None,
            enable_thinking: request.enable_thinking,
            bos_token: "",
        },
    )
    .map_err(|e| format!("{e:#}"))
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
    if let Some(note) = tool_choice_note(tool_choice) {
        directive.push('\n');
        directive.push_str(&note);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The history rig hands us after one completed tool call: the assistant
    /// turn that made it (no text, `content: null`) and the result.
    fn tool_history() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": "Which vendor?"}),
            json!({
                "role": "assistant",
                "content": Value::Null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "schema", "arguments": "{\"kind\":\"asset\"}"},
                }],
            }),
            json!({"role": "tool", "tool_call_id": "call_1", "content": "{\"fields\":[]}"}),
        ]
    }

    fn request(tools: Option<&str>) -> PreparedRequest {
        PreparedRequest {
            messages_json: "[]".to_string(),
            tools_json: tools.map(str::to_string),
            tool_choice: None,
            json_schema: None,
            enable_thinking: None,
            #[cfg(feature = "mtmd")]
            images: Vec::new(),
        }
    }

    #[test]
    fn a_tool_free_turn_does_not_take_the_native_path() {
        let plain = vec![json!({"role": "user", "content": "Hi"})];
        assert!(!involves_tools(&request(None), &plain));
    }

    #[test]
    fn history_alone_keeps_the_turn_on_the_native_path() {
        // The follow-up turn carries no `tools`, but the assistant's earlier
        // call still has to survive into the prompt.
        assert!(involves_tools(&request(None), &tool_history()));
    }

    #[test]
    fn the_native_path_keeps_the_assistant_tool_call() {
        let messages = structured_messages(&tool_history());
        let assistant = &messages[2];
        let call = &assistant["tool_calls"][0];
        assert_eq!(call["function"]["name"], json!("schema"));
        // Parsed to an object, not left as an encoded string, so `| tojson`
        // renders `{"kind": "asset"}` rather than a quoted blob.
        assert_eq!(call["function"]["arguments"], json!({"kind": "asset"}));
        assert_eq!(messages[3]["role"], json!("tool"));
        assert_eq!(messages[3]["tool_call_id"], json!("call_1"));
    }

    #[test]
    fn a_qwen_shaped_template_renders_the_call_and_its_result() {
        // The fragment of Qwen3's template that reads the fields path 1 drops.
        let tmpl = "{%- for m in messages -%}\
                    {%- if m.role == 'tool' -%}\
                    [tool_response:{{ m.content }}]\
                    {%- elif m.tool_calls -%}\
                    {%- for c in m.tool_calls -%}\
                    [call:{{ c.function.name }}:{{ c.function.arguments | tojson }}]\
                    {%- endfor -%}\
                    {%- endif -%}\
                    {%- endfor -%}";
        let messages = structured_messages(&tool_history());
        let out = crate::jinja::render_chat_template(
            tmpl,
            &messages,
            &crate::jinja::TemplateVars {
                tools: None,
                enable_thinking: None,
                bos_token: "",
            },
        )
        .unwrap();
        assert_eq!(
            out,
            // `tojson` spaces its output as Python's `json.dumps` does.
            "[call:schema:{\"kind\": \"asset\"}][tool_response:{\"fields\":[]}]"
        );
    }

    #[test]
    fn the_fallback_flatten_keeps_tool_traffic_as_text() {
        // Regression guard: this used to yield an empty assistant turn and a
        // `role: "tool"` message no ChatML-family model was trained on.
        let flat = flatten_messages(&tool_history());
        assert_eq!(
            flat[2],
            (
                "assistant".to_string(),
                "<tool_call>\n{\"name\": \"schema\", \"arguments\": {\"kind\":\"asset\"}}\n\
                 </tool_call>"
                    .to_string()
            )
        );
        assert_eq!(
            flat[3],
            (
                "user".to_string(),
                "<tool_response>\n{\"fields\":[]}\n</tool_response>".to_string()
            )
        );
    }

    #[test]
    fn the_fallback_flatten_keeps_text_alongside_a_call() {
        let msg = vec![json!({
            "role": "assistant",
            "content": "Let me look.",
            "tool_calls": [{"function": {"name": "list", "arguments": "{}"}}],
        })];
        let flat = flatten_messages(&msg);
        assert_eq!(
            flat[0].1,
            "Let me look.\n<tool_call>\n{\"name\": \"list\", \"arguments\": {}}\n</tool_call>"
        );
    }

    #[test]
    fn a_tool_choice_reaches_the_native_path_as_prose() {
        let mut messages = structured_messages(&tool_history());
        merge_system_note(&mut messages, &tool_choice_note(Some("required")).unwrap());
        assert_eq!(
            messages[0]["content"],
            json!("Be brief.\n\nTool choice: You must call at least one tool in your response.")
        );
        assert!(tool_choice_note(Some("auto")).is_none());
        assert!(tool_choice_note(None).is_none());
    }

    /// The real templates, lifted verbatim out of the GGUFs. A hand-written
    /// stub proves only that the code renders something; these are what the
    /// models actually ship, and what the native path has to survive.
    const QWEN3: &str = include_str!("../tests/fixtures/qwen3.6-chat-template.jinja");
    const GEMMA4: &str = include_str!("../tests/fixtures/gemma4-chat-template.jinja");

    fn render_native(template: &str, enable_thinking: Option<bool>) -> String {
        let tools = json!([{
            "type": "function",
            "function": {
                "name": "schema",
                "description": "Describe an entity kind.",
                "parameters": {"type": "object", "properties": {"kind": {"type": "string"}}},
            },
        }]);
        crate::jinja::render_chat_template(
            template,
            &structured_messages(&tool_history()),
            &crate::jinja::TemplateVars {
                tools: Some(&tools),
                enable_thinking,
                bos_token: "",
            },
        )
        .expect("a shipped chat template must render")
    }

    #[test]
    fn qwen3s_own_template_renders_the_tool_call_it_made() {
        let out = render_native(QWEN3, None);
        // The tool is advertised natively, in the model's own <tools> block.
        assert!(
            out.contains("\"name\": \"schema\""),
            "tools missing:\n{out}"
        );
        // Qwen3.6 asks for the `<function=…><parameter=…>` form rather than
        // the JSON one, and renders history the same way.
        assert!(
            out.contains(
                "<tool_call>\n<function=schema>\n<parameter=kind>\nasset\n</parameter>\n\
                 </function>\n</tool_call>"
            ),
            "the assistant's own tool call is missing:\n{out}"
        );
        // And its result comes back as a user-role <tool_response>, which is
        // what Qwen was trained on — not a bare `role: "tool"` turn.
        assert!(
            out.contains("<tool_response>\n{\"fields\":[]}\n</tool_response>"),
            "tool response missing:\n{out}"
        );
    }

    #[test]
    fn qwen3s_rendered_call_parses_back_to_the_call_that_made_it() {
        // The loop that matters: what the template emits for a past call is
        // what `parse_tool_calls` reads out of the next completion. If these
        // two ever disagree the model is shown one protocol and answers in
        // another. The template's own instruction block contributes an
        // `example_function_name` call ahead of ours.
        let out = render_native(QWEN3, None);
        let calls =
            crate::parsing::parse_xml_tool_calls(&out, None).expect("no tool call parsed back");
        assert_eq!(
            calls.last().expect("at least one call"),
            &("schema".to_string(), json!({"kind": "asset"}))
        );
    }

    #[test]
    fn gemma4s_own_template_renders_the_tool_call_it_made() {
        let out = render_native(GEMMA4, None);
        assert!(out.contains("schema"), "tools missing:\n{out}");
        // Gemma answers in its own DSL, which `parse_gemma_tool_calls` reads.
        assert!(
            out.contains("<|tool_call>"),
            "the assistant's own tool call is missing:\n{out}"
        );
    }

    #[test]
    fn an_unspecified_thinking_preference_leaves_qwen_reasoning_on() {
        // Qwen3.6 forces non-thinking mode by prefilling an empty
        // `<think></think>` and otherwise opens a real one. Only an explicit
        // `false` may close it — an absent preference must not.
        assert!(
            render_native(QWEN3, None).ends_with("<think>\n"),
            "an unasked-for preference switched reasoning off"
        );
        assert!(render_native(QWEN3, Some(false)).ends_with("<think>\n\n</think>\n\n"));
        assert!(render_native(QWEN3, Some(true)).ends_with("<think>\n"));
    }

    #[test]
    fn a_tool_choice_note_creates_a_system_turn_when_there_is_none() {
        let mut messages = vec![json!({"role": "user", "content": "Hi"})];
        merge_system_note(&mut messages, "Tool choice: none");
        assert_eq!(messages[0]["role"], json!("system"));
        assert_eq!(messages[1]["role"], json!("user"));
    }
}
