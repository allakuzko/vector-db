mod store;
mod embedder;

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tracing::{error, info};

use embedder::Embedder;
use store::{SearchResult, VectorStore};

// --- Помилки ---

struct AppError(anyhow::Error);

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        error!("Error: {}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": self.0.to_string()
            })),
        )
            .into_response()
    }
}

impl<E: Into<anyhow::Error>> From<E> for AppError {
    fn from(e: E) -> Self {
        AppError(e.into())
    }
}

type AppResult<T> = Result<Json<T>, AppError>;

// --- Структури ---

#[derive(Deserialize)]
struct InsertRequest {
    text: String,
}

#[derive(Serialize)]
struct InsertResponse {
    id: String,
    message: String,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    top_k: Option<usize>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResult>,
    total_docs: usize,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    total_docs: usize,
}

// --- Стан ---

struct AppState {
    store: Mutex<VectorStore>,
    embedder: Embedder,
}

// --- Main ---

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    info!("Initializing embedder...");
    let embedder = Embedder::new().await.expect("Failed to load embedder");
    info!("Embedder ready!");

    let state = Arc::new(AppState {
        store: Mutex::new(VectorStore::new()),
        embedder,
    });

    let app = Router::new()
        .route("/insert", post(insert))
        .route("/search", post(search))
        .route("/health", get(health))
        .with_state(state);

    let addr = "0.0.0.0:3000";
    info!("Server running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind");

    axum::serve(listener, app).await.expect("Server error");
}

// --- Handlers ---

async fn health(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
) -> Json<HealthResponse> {
    let total_docs = state.store.lock().unwrap().len();
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        total_docs,
    })
}

async fn insert(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<InsertRequest>,
) -> AppResult<InsertResponse> {
    if payload.text.trim().is_empty() {
        return Err(AppError(anyhow::anyhow!("Text cannot be empty")));
    }

    let embedding = state.embedder.embed(&payload.text)?;

    let id = state
        .store
        .lock()
        .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?
        .insert(payload.text.clone(), embedding);

    info!(id = %id, text = %payload.text, "Document inserted");

    Ok(Json(InsertResponse {
        id,
        message: "Document inserted successfully".to_string(),
    }))
}

async fn search(
    axum::extract::State(state): axum::extract::State<Arc<AppState>>,
    Json(payload): Json<SearchRequest>,
) -> AppResult<SearchResponse> {
    if payload.query.trim().is_empty() {
        return Err(AppError(anyhow::anyhow!("Query cannot be empty")));
    }

    let top_k = payload.top_k.unwrap_or(5).min(20);
    let total_docs = state.store.lock().unwrap().len();

    let query_embedding = state.embedder.embed(&payload.query)?;

    let results = state
        .store
        .lock()
        .map_err(|e| anyhow::anyhow!("Lock error: {}", e))?
        .search(&query_embedding, top_k);

    info!(
        query = %payload.query,
        results = %results.len(),
        "Search completed"
    );

    Ok(Json(SearchResponse { results, total_docs }))
}