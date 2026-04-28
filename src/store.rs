use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: String,
    pub text: String,
    pub score: f32,
}

pub struct VectorStore {
    documents: HashMap<String, Document>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
        }
    }

    pub fn insert(&mut self, text: String, embedding: Vec<f32>) -> String {
        let id = Uuid::new_v4().to_string();
        let doc = Document {
            id: id.clone(),
            text,
            embedding,
        };
        self.documents.insert(id.clone(), doc);
        id
    }

    pub fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<SearchResult> {
        let mut scores: Vec<(&Document, f32)> = self
            .documents
            .values()
            .map(|doc| {
                let score = cosine_similarity(query_embedding, &doc.embedding);
                (doc, score)
            })
            .collect();

        // Сортуємо за score від найбільшого
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        scores
            .into_iter()
            .take(top_k)
            .map(|(doc, score)| SearchResult {
                id: doc.id.clone(),
                text: doc.text.clone(),
                score,
            })
            .collect()
    }

    pub fn len(&self) -> usize {
        self.documents.len()
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}
