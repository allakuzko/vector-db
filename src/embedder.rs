use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::Tokenizer;

pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl Embedder {
    pub async fn new() -> Result<Self> {
        let device = Device::Cpu;
        let model_path = std::path::Path::new("model");

        tracing::info!("Loading model from local files...");

        let config: Config = serde_json::from_str(
            &std::fs::read_to_string(model_path.join("config.json"))?
        )?;

        let tokenizer = Tokenizer::from_file(model_path.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("Tokenizer error: {}", e))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_path.join("model.safetensors")],
                candle_core::DType::F32,
                &device,
            )?
        };

        let model = BertModel::load(vb, &config)?;

        tracing::info!("Model loaded successfully!");

        Ok(Self { model, tokenizer, device })
    }

    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;

        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let type_ids = encoding.get_type_ids();

        let input_ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        let attention_mask = Tensor::new(mask, &self.device)?.unsqueeze(0)?;
        let token_type_ids = Tensor::new(type_ids, &self.device)?.unsqueeze(0)?;

        let output = self.model.forward(
            &input_ids,
            &token_type_ids,
            Some(&attention_mask),
        )?;

        tracing::info!("Output shape: {:?}", output.shape());

        // Просто flatten і normalize без складних операцій
        let output = output.squeeze(0)?;
        tracing::info!("After squeeze: {:?}", output.shape());

        let cls = output.get(0)?;
        tracing::info!("CLS shape: {:?}", cls.shape());

        let norm = cls.sqr()?.sum_all()?.sqrt()?;
        let normalized = cls.broadcast_div(&norm)?;
        Ok(normalized.to_vec1::<f32>()?)
}
}

fn mean_pooling(output: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
    // output: [1, seq_len, hidden] -> беремо середнє по seq_len
    let (_batch, seq_len, _hidden) = output.dims3()?;
    
    let mask = attention_mask
        .to_dtype(candle_core::DType::F32)?
        .reshape((1, seq_len, 1))?;
    
    let sum = (output * &mask)?.sum(1)?;
    let count = mask.sum(1)?;
    Ok((sum / count)?)
}

fn normalize(tensor: &Tensor) -> Result<Tensor> {
    // tensor: [1, hidden]
    let norm = tensor.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok((tensor / norm)?)
}
