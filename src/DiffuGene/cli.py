#!/usr/bin/env python
"""
DiffuGene Command Line Interface

This provides a unified entry point for all DiffuGene functionality.
"""

import sys
import argparse
from typing import List

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog='diffugene',
        description='DiffuGene: Genetic Diffusion Modeling Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available commands:
  pipeline    Run the complete end-to-end pipeline
  
Individual steps:
  data-prep   Data preparation and LD block inference
  block-pca   Block-wise PCA embedding
  vae-train   VAE training on block embeddings
  vae-infer   VAE inference to generate latents
  diff-train  Diffusion model training
  diff-gen    Generate samples using diffusion model
  
Examples:
  # Run complete pipeline
  diffugene pipeline
  
  # Run pipeline with custom config
  diffugene pipeline --config my_config.yaml
  
  # Run specific steps
  diffugene pipeline --steps data_prep block_embed
  
  # Run individual commands
  diffugene vae-train --model-save-path models/vae.pt --epochs 100
        """
    )
    
    parser.add_argument(
        'command',
        choices=[
            'pipeline', 
            'data-prep', 'block-pca', 
            'vae-train', 'vae-infer',
            'diff-train', 'diff-gen'
        ],
        help='Command to run'
    )
    
    # Parse known args to handle subcommand arguments
    args, remaining = parser.parse_known_args()
    
    # Route to appropriate module
    if args.command == 'pipeline':
        from .pipeline import main as pipeline_main
        # Reconstruct sys.argv for the pipeline
        sys.argv = ['pipeline'] + remaining
        pipeline_main()
        
    elif args.command == 'data-prep':
        from .block_embed.run import main as data_prep_main
        sys.argv = ['data-prep'] + remaining
        data_prep_main()
        
    elif args.command == 'block-pca':
        from .block_embed.fit_pca import main as pca_main
        sys.argv = ['block-pca'] + remaining
        pca_main()
        
    elif args.command == 'vae-train':
        from .joint_embed.train import main as vae_train_main
        sys.argv = ['vae-train'] + remaining
        vae_train_main()
        
    elif args.command == 'vae-infer':
        from .joint_embed.infer import main as vae_infer_main
        sys.argv = ['vae-infer'] + remaining
        vae_infer_main()
        
    elif args.command == 'diff-train':
        from .diffusion.train import main as diff_train_main
        sys.argv = ['diff-train'] + remaining
        diff_train_main()
        
    elif args.command == 'diff-gen':
        from .diffusion.generate import main as diff_gen_main
        sys.argv = ['diff-gen'] + remaining
        diff_gen_main()
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
