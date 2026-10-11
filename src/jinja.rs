//! Jinja chat-template rendering: the only path that can express a
//! tool-calling turn, and the fallback for models llama.cpp cannot template.
//!
//! `llama_chat_apply_template` is not a jinja engine. It sniffs the template
//! string for a handful of well-known shapes (ChatML, Llama-3, Mistral, …) and
//! returns `-1` for anything it does not recognise. It also accepts only
//! `(role, content)` pairs, so template variables like `enable_thinking` and
//! structured `tools` have nowhere to go — `llama-cpp-2` 0.1.147 removed the
//! `chat_template_kwargs` plumbing that used to forward them.
//!
//! That limit costs two different things:
//!
//! - **Tool calls.** `tool_calls` is a sibling of `content`, so a
//!   `(role, content)` pair cannot carry one. Flattening an assistant turn that
//!   called a tool erases the call, and the model is shown a tool result it has
//!   no record of requesting. Rendering the template here is the only way to
//!   put the call back — see [`crate::prompt::build_prompt`].
//! - **Bespoke formats.** Models shipping a template llama.cpp does not
//!   recognise fail that path outright and, before this module existed, were
//!   silently prompted with ChatML — a format they were never trained on.
//!   Gemma-4 is one: its template renders `<|turn>role` turns and gates
//!   reasoning behind `enable_thinking`.
//!
//! For a tool-free turn whose template llama.cpp does recognise, its applier
//! remains the primary path and the prompt is byte-for-byte unchanged.

use minijinja::value::Value as JValue;
use minijinja::{Environment, Error, ErrorKind, State};
use serde_json::Value;

/// Python-style methods that real chat templates call but minijinja does not
/// implement natively. Upstream ships a broad set in `minijinja-contrib`'s
/// `pycompat` module; the set below is what the templates we render actually
/// call, so we implement them here rather than take a second dependency.
///
/// Kept in step with the fixtures under `tests/fixtures/*.jinja` — those are
/// verbatim copies of shipped templates, and a method missing from here fails
/// the whole render, which drops the turn to a lossy fallback path.
fn unknown_method(
    _state: &mut State,
    value: &JValue,
    method: &str,
    args: &[JValue],
) -> Result<JValue, Error> {
    /// The string a str-method was called on, or an `InvalidOperation` naming
    /// the culprit — a template bug is far easier to read than a type error.
    fn text<'a>(value: &'a JValue, method: &str) -> Result<&'a str, Error> {
        value.as_str().ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidOperation,
                format!("{method}() called on a non-string"),
            )
        })
    }

    /// The single string argument of a str-method, if it has one.
    fn arg(args: &[JValue]) -> Option<&str> {
        args.first().and_then(|a| a.as_str())
    }

    match method {
        // `message.get('tool_calls')` / `.get(key, default)`. Jinja2 returns
        // the default (None when omitted) for a missing key rather than
        // raising, which templates rely on to probe optional fields.
        "get" => {
            let key = args
                .first()
                .ok_or_else(|| Error::new(ErrorKind::MissingArgument, "get() requires a key"))?;
            let default = args.get(1).cloned().unwrap_or_else(|| JValue::from(()));
            match value.get_item(key) {
                Ok(found) if !found.is_undefined() => Ok(found),
                _ => Ok(default),
            }
        }
        // `text.split('<channel|>')`, and the no-argument whitespace form.
        "split" => {
            let text = text(value, method)?;
            let parts: Vec<JValue> = match arg(args) {
                Some(sep) => text.split(sep).map(JValue::from).collect(),
                None => text.split_whitespace().map(JValue::from).collect(),
            };
            Ok(JValue::from(parts))
        }
        // `content.startswith('<think>')` / `.endswith('</tool_call>')`.
        // Python returns False for a non-string argument rather than raising.
        "startswith" => Ok(JValue::from(arg(args).is_some_and(|prefix| {
            text(value, method).is_ok_and(|t| t.starts_with(prefix))
        }))),
        "endswith" => Ok(JValue::from(arg(args).is_some_and(|suffix| {
            text(value, method).is_ok_and(|t| t.ends_with(suffix))
        }))),
        // `content.lstrip('\n')` / `.rstrip()`. Python strips *any* of the
        // characters in the argument, not the argument as a substring; with no
        // argument it strips whitespace.
        "strip" | "lstrip" | "rstrip" => {
            let text = text(value, method)?;
            let cut: &dyn Fn(&str) -> &str = &|s: &str| match arg(args) {
                Some(chars) => match method {
                    "lstrip" => s.trim_start_matches(|c| chars.contains(c)),
                    "rstrip" => s.trim_end_matches(|c| chars.contains(c)),
                    _ => s.trim_matches(|c| chars.contains(c)),
                },
                None => match method {
                    "lstrip" => s.trim_start(),
                    "rstrip" => s.trim_end(),
                    _ => s.trim(),
                },
            };
            Ok(JValue::from(cut(text)))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

/// Variables a chat template expects alongside the message list.
pub(crate) struct TemplateVars<'a> {
    /// The OpenAI-shaped `tools` array, handed to the template so it can render
    /// its own tool-call protocol. `None` closes the template's tool block and
    /// leaves tools to the portable directive `build_prompt` injects instead —
    /// see [`render_chat_template`] for which path picks which.
    pub tools: Option<&'a Value>,
    /// Forwarded from the request's `additional_params` (`{"thinking": bool}`).
    /// `None` reaches the template as *undefined* rather than `false`, so a
    /// template that reasons by default keeps doing so.
    pub enable_thinking: Option<bool>,
    /// The model's BOS piece. Templates emit it explicitly via `bos_token`;
    /// tokenization adds BOS separately, so an empty string here avoids
    /// doubling it.
    pub bos_token: &'a str,
}

/// Render `template` with minijinja.
///
/// Called two ways, and the difference is entirely in `messages` and `tools`:
///
/// - **Tool turns** get the structured messages — `tool_calls` intact,
///   `role: "tool"` preserved — plus a native `tools` array, because nothing
///   else can represent an assistant turn that called a tool. The model then
///   answers in its own dialect, which `parse_tool_calls` reads: the portable
///   `<tool_call>` JSON protocol *and* Gemma-4's `<|tool_call>call:name{k:v}`
///   DSL, added in 0.4.1. Before that parser existed a native dialect yielded
///   no tool calls at all, which is why this path used to be avoided.
/// - **Everything else** gets flattened `{role, content}` pairs and no `tools`,
///   so the turn format is the only thing that changes versus ChatML.
///
/// **Media** is flattened on both: an image arrives as a `media_marker` part
/// whose text is llama.cpp's marker, and inlining it into the content string
/// keeps markers and bitmaps balanced. A template iterating raw content parts
/// drops the type it does not recognise, and mtmd then rejects the prompt for
/// having fewer markers than bitmaps.
pub(crate) fn render_chat_template(
    template: &str,
    messages: &[Value],
    vars: &TemplateVars<'_>,
) -> Result<String, Error> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method);
    // Chat templates are whitespace-sensitive: they control spacing with
    // explicit `{%-` / `-%}` markers and rely on everything else surviving
    // verbatim. Keep minijinja from trimming on its own.
    env.set_syntax(
        minijinja::syntax::SyntaxConfig::builder()
            .keep_trailing_newline(true)
            .build()?,
    );
    env.add_template("chat", template)?;
    let tmpl = env.get_template("chat")?;

    tmpl.render(minijinja::context! {
        messages => JValue::from(minijinja::value::Serde(messages)),
        // When absent, explicitly none rather than undefined: templates branch
        // on `tools` in boolean context, and none is unambiguously falsy.
        tools => match vars.tools {
            Some(tools) => JValue::from(minijinja::value::Serde(tools)),
            None => JValue::from(()),
        },
        // When absent, undefined rather than none — templates test
        // `enable_thinking is defined` to tell "off" from "unspecified".
        enable_thinking => match vars.enable_thinking {
            Some(on) => JValue::from(on),
            None => JValue::UNDEFINED,
        },
        // Templates that render reasoning back into history gate it on this;
        // leave it off so prior turns' thoughts are not replayed.
        preserve_thinking => false,
        add_generation_prompt => true,
        bos_token => vars.bos_token,
        eos_token => "",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn msgs() -> Vec<Value> {
        vec![
            json!({"role": "system", "content": "Be brief."}),
            json!({"role": "user", "content": "Hi"}),
        ]
    }

    fn vars<'a>(thinking: Option<bool>) -> TemplateVars<'a> {
        TemplateVars {
            tools: None,
            enable_thinking: thinking,
            bos_token: "",
        }
    }

    fn vars_with_tools<'a>(tools: &'a Value) -> TemplateVars<'a> {
        TemplateVars {
            tools: Some(tools),
            enable_thinking: None,
            bos_token: "",
        }
    }

    #[test]
    fn renders_a_chatml_style_template() {
        // Non-trimming `{% endfor %}` / `{% endif %}` so the newlines the
        // template emits survive — chat templates are whitespace-sensitive.
        let tmpl = "{%- for m in messages -%}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}{%- if add_generation_prompt -%}<|im_start|>assistant\n{% endif %}";
        let out = render_chat_template(tmpl, &msgs(), &vars(None)).unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn enable_thinking_reaches_the_template() {
        let tmpl = "{%- set enable_thinking = enable_thinking | default(false) -%}\
                    {%- if enable_thinking -%}<|think|>{%- endif -%}done";
        let on = render_chat_template(tmpl, &msgs(), &vars(Some(true))).unwrap();
        let off = render_chat_template(tmpl, &msgs(), &vars(Some(false))).unwrap();
        assert_eq!(on, "<|think|>done");
        assert_eq!(off, "done");
    }

    #[test]
    fn an_unspecified_thinking_preference_leaves_the_template_default_alone() {
        // Qwen3's shape: reasoning stays on unless explicitly switched off, so
        // `None` must arrive undefined rather than as `false`.
        let tmpl = "{%- if enable_thinking is defined and enable_thinking is false -%}\
                    off{%- else -%}on{%- endif -%}";
        assert_eq!(
            render_chat_template(tmpl, &msgs(), &vars(None)).unwrap(),
            "on"
        );
        assert_eq!(
            render_chat_template(tmpl, &msgs(), &vars(Some(false))).unwrap(),
            "off"
        );
    }

    #[test]
    fn tools_are_absent_unless_the_caller_passes_them() {
        // A tool-free turn keeps the template's tool block closed; tools then
        // reach the model through the portable `<tool_call>` directive that
        // `build_prompt` injects into the system message.
        let tmpl = "{%- if tools -%}native{%- else -%}absent{%- endif -%}";
        let out = render_chat_template(tmpl, &msgs(), &vars(None)).unwrap();
        assert_eq!(out, "absent");
    }

    #[test]
    fn a_tool_turn_hands_the_template_its_native_tools() {
        let tools = json!([{"type": "function", "function": {"name": "list"}}]);
        let tmpl = "{%- for t in tools -%}{{ t.function.name }}{%- endfor -%}";
        let out = render_chat_template(tmpl, &msgs(), &vars_with_tools(&tools)).unwrap();
        assert_eq!(out, "list");
    }

    #[test]
    fn get_method_returns_default_for_missing_key() {
        // The `.get()` pycompat shim: present key, missing key, explicit default.
        let tmpl = "{{ messages[0].get('role') }}|{{ messages[0].get('nope') }}|{{ messages[0].get('nope', 'fb') }}";
        let out = render_chat_template(tmpl, &msgs(), &vars(None)).unwrap();
        // minijinja prints none the way Python's Jinja2 does.
        assert_eq!(out, "system|None|fb");
    }

    #[test]
    fn split_method_splits_on_a_separator() {
        let tmpl = "{%- for p in 'a<channel|>b'.split('<channel|>') -%}[{{ p }}]{%- endfor -%}";
        let out = render_chat_template(tmpl, &msgs(), &vars(None)).unwrap();
        assert_eq!(out, "[a][b]");
    }

    #[test]
    fn supports_macros_namespaces_and_dictsort() {
        // The constructs Gemma-4's template leans on hardest.
        let tmpl = "{%- macro emit(k, v) -%}{{ k }}={{ v }};{%- endmacro -%}\
                    {%- set ns = namespace(n=0) -%}\
                    {%- for k, v in {'b': 2, 'a': 1} | dictsort -%}\
                    {%- set ns.n = ns.n + 1 -%}{{ emit(k, v) }}\
                    {%- endfor -%}count={{ ns.n }}";
        let out = render_chat_template(tmpl, &msgs(), &vars(None)).unwrap();
        assert_eq!(out, "a=1;b=2;count=2");
    }

    #[test]
    fn reports_an_error_for_a_broken_template() {
        assert!(render_chat_template("{% for %}", &msgs(), &vars(None)).is_err());
    }
}
