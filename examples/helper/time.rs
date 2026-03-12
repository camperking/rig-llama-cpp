use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Deserialize, Serialize)]
pub struct GetCurrentTimeArgs {}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct TimeError(String);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GetCurrentTime;

impl Tool for GetCurrentTime {
    const NAME: &'static str = "get_time";

    type Error = TimeError;
    type Args = GetCurrentTimeArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_time".to_string(),
            description: "Gibt das aktuelle Datum und die aktuelle Uhrzeit im UTC-Format zurück (YYYY-MM-DD HH:MM:SS UTC)".to_string(),
            parameters: json!({}),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        use chrono::Utc;
        Ok(format!(
            "Current time: {}",
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ))
    }
}
