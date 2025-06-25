# DiffuGene: Using Latent Diffusion Models to Generate Synthetic Genotypes

DiffuGene is an end-to-end conditional latent diffusion framework for scalable genetic data synthesis. It learns a compact joint embedding of genome-wide SNP blocks via a lightweight VAE, then trains a U-Net-based diffusion model to denoise and sample in that latent space under user-specified covariates. At generation time it maps sampled latent codes back through the learned PCA projections to produce high-fidelity, realistic synthetic genotypes.

## Pipeline Overview

The DiffuGene pipeline consists of 5 main steps:

1. **Data Preparation & LD Block Inference** - Process raw genetic data and infer linkage disequilibrium blocks
2. **Block-wise PCA Embedding** - Apply Principal Component Analysis to each LD block  
3. **Joint VAE Embedding** - Train a Variational Autoencoder on stacked block embeddings
4. **Diffusion Model Training** - Train a UNet-based diffusion model on VAE latents
5. **Sample Generation** - Generate new synthetic genetic samples

## Quick Start

### Installation

```bash
# Clone the repository
cd /path/to/DiffuGene

# Install the package in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt
```

### Configuration

The pipeline is controlled by a YAML configuration file. You can use the default config or create your own:

```bash
# View default configuration
cat src/DiffuGene/config/pipeline.yaml

# Copy and modify for your needs
cp src/DiffuGene/config/pipeline.yaml my_config.yaml
```

### Running the Pipeline

#### Option 1: Shell Script (Recommended)
```bash
# Run with custom config
./scripts/run_pipeline.sh --config my_config.yaml

# Run specific steps only
./scripts/run_pipeline.sh --steps data_prep,block_embed

# Force rerun all steps
./scripts/run_pipeline.sh --force-rerun

# Use specific Python environment
./scripts/run_pipeline.sh --env my_conda_env
```

#### Option 2: Python Module
```bash
# Run with custom config
python -m DiffuGene.pipeline --config my_config.yaml

# Run specific steps
python -m DiffuGene.pipeline --steps data_prep block_embed
```

#### Option 3: CLI Interface
```bash
# Run complete pipeline
diffugene pipeline

# Run individual steps
diffugene vae-train --model-save-path models/vae.pt --epochs 100
diffugene diff-train --model-output-path models/diffusion.pth
```


## Data Flow

```
Raw Genetic Data (.bed/.bim/.fam)
       ↓
LD Blocks (.blocks.det) + Recoded Data (.raw)
       ↓  
Blockwise PCA Embeddings (.pt files)
       ↓
VAE Joint Embeddings (.pt file)
       ↓
Train Diffusion Model (.pth checkpoint)
       ↓
Generate Samples (.pt file)
```

## Output Files

- **PCA Embeddings**: `data/haploblocks_embeddings_PCA/`
- **VAE Model**: `models/joint_embed/vae_model.pt`
- **VAE Latents**: `data/VAE_embeddings/`
- **Diffusion Model**: `models/diffusion/ddpm_checkpoint.pth`
- **Generated Samples**: `data/generated_samples/`

## Monitoring and Logging

Logs are automatically generated with timestamps and can be controlled via the configuration:

```yaml
logging:
  level: "INFO"
  log_file: "models/pipeline.log"
  console_output: true
```
