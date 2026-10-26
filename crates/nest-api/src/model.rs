//! Composition HTTP transport; lifecycle decisions live in nest-place.
use crate::{Api, ApiError};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use serde::{Deserialize, Serialize};

pub(crate) fn router() -> Router<Api> {
    Router::new()
        .route("/v1/model", get(list))
        .route("/v1/model/cache/status", get(cache_status))
        .route("/v1/model/cache/reset", post(reset_all))
        .route("/v1/model/{name}", get(status).put(put))
        .route("/v1/model/{name}/prepare", post(prepare))
        .route("/v1/model/{name}/resolve", get(resolve))
        .route("/v1/model/{name}/qualify", post(qualify))
        .route("/v1/model/{name}/cache/reset", post(reset))
        .route("/v1/model/{name}/cache/reset-cancel", post(cancel_reset))
        .route("/v1/model/{name}/use", post(acquire))
        .route("/v1/model/usage/release", post(release))
        .route("/v1/model/{name}/jobs/stop", post(stop))
        .route("/v1/model/{name}/jobs/recover", post(recover))
        .layer(DefaultBodyLimit::max(
            nest_model::worker::MAX_REQUEST_BYTES as usize,
        ))
}

type ModelResponse = Result<Response, ApiError>;

static RESPONSE_ENCODER: tokio::sync::Semaphore = tokio::sync::Semaphore::const_new(1);

async fn response<T: Serialize + Send + 'static>(
    value: nest_types::NestResult<T>,
) -> ModelResponse {
    let value = value?;
    let internal = |error: String| ApiError(axum::http::StatusCode::INTERNAL_SERVER_ERROR, error);
    let permit = RESPONSE_ENCODER
        .acquire()
        .await
        .map_err(|e| internal(e.to_string()))?;
    let bytes = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        serde_json::to_vec(&value)
    })
    .await
    .map_err(|e| internal(e.to_string()))?
    .map_err(|e| internal(e.to_string()))?;
    Ok((
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    )
        .into_response())
}

async fn put(
    State(api): State<Api>,
    Path(name): Path<String>,
    body: axum::body::Bytes,
) -> ModelResponse {
    response(api.composition.put_json(&name, body).await).await
}
async fn list(State(api): State<Api>) -> ModelResponse {
    response(api.composition.list().await).await
}
async fn cache_status(State(api): State<Api>) -> ModelResponse {
    response(api.composition.cache_status().await).await
}
async fn status(State(api): State<Api>, Path(name): Path<String>) -> ModelResponse {
    response(api.composition.status(&name).await).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Prepare {
    #[serde(default)]
    wait: bool,
}
async fn prepare(
    State(api): State<Api>,
    Path(name): Path<String>,
    Json(body): Json<Prepare>,
) -> ModelResponse {
    response(api.composition.prepare(&name, body.wait).await).await
}
async fn resolve(State(api): State<Api>, Path(name): Path<String>) -> ModelResponse {
    response(api.composition.resolve(&name).await).await
}
async fn qualify(State(api): State<Api>, Path(name): Path<String>) -> ModelResponse {
    response(api.composition.qualify(&name).await).await
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Reset {
    #[serde(default)]
    when_idle: bool,
    #[serde(default)]
    dry_run: bool,
}
async fn reset(
    State(api): State<Api>,
    Path(name): Path<String>,
    Json(body): Json<Reset>,
) -> ModelResponse {
    response(
        api.composition
            .reset(&name, body.when_idle, body.dry_run)
            .await,
    )
    .await
}
async fn reset_all(State(api): State<Api>, Json(body): Json<Reset>) -> ModelResponse {
    response(
        api.composition
            .reset_all(body.when_idle, body.dry_run)
            .await,
    )
    .await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CancelReset {
    epoch: u64,
    reset: u64,
}
async fn cancel_reset(
    State(api): State<Api>,
    Path(name): Path<String>,
    Json(body): Json<CancelReset>,
) -> ModelResponse {
    response(
        api.composition
            .cancel_reset(&name, body.epoch, body.reset)
            .await,
    )
    .await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Acquire {
    lease: String,
}
async fn acquire(
    State(api): State<Api>,
    Path(name): Path<String>,
    Json(body): Json<Acquire>,
) -> ModelResponse {
    response(api.composition.acquire(&name, &body.lease).await).await
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Release {
    key: String,
    epoch: u64,
    lease: String,
}
async fn release(State(api): State<Api>, Json(body): Json<Release>) -> ModelResponse {
    response(
        api.composition
            .release(&body.key, body.epoch, &body.lease)
            .await,
    )
    .await
}
async fn stop(State(api): State<Api>, Path(name): Path<String>) -> ModelResponse {
    response(api.composition.stop(&name).await).await
}
async fn recover(State(api): State<Api>, Path(name): Path<String>) -> ModelResponse {
    response(api.composition.recover(&name).await).await
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SerializationThread(std::thread::ThreadId);
    impl Serialize for SerializationThread {
        fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            assert_ne!(
                std::thread::current().id(),
                self.0,
                "response serialization occupied the control executor"
            );
            serializer.serialize_str("proof")
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn response_encoding_leaves_the_control_executor() {
        let result = response(Ok(SerializationThread(std::thread::current().id())))
            .await
            .unwrap_or_else(|e| panic!("{}", e.1));
        let bytes = axum::body::to_bytes(result.into_body(), 1024)
            .await
            .unwrap();
        assert_eq!(bytes.as_ref(), br#""proof""#);
    }
}
