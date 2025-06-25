#!/bin/bash
#
# DiffuGene Pipeline Runner
# 
# This script provides a convenient wrapper for running the DiffuGene pipeline
# with proper environment setup and error handling.
#

set -e  # Exit on any error

# Default values
CONFIG_FILE=""
STEPS=""
FORCE_RERUN=false
PYTHON_ENV=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[DiffuGene]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
DiffuGene Pipeline Runner

Usage: $0 [OPTIONS]

Options:
    -c, --config FILE       Configuration YAML file (default: use package default)
    -s, --steps STEPS       Comma-separated list of steps to run
                           (data_prep,block_embed,joint_embed,diffusion,generation)
    -f, --force-rerun       Force rerun all steps even if outputs exist
    -e, --env ENV           Python environment to activate (conda/venv name)
    -h, --help              Show this help message

Examples:
    # Run complete pipeline with default config
    $0
    
    # Run with custom config file
    $0 --config my_config.yaml
    
    # Run specific steps only
    $0 --steps data_prep,block_embed
    
    # Force rerun with conda environment
    $0 --force-rerun --env diffugene_env

Available Steps:
    data_prep     - Data preparation and LD block inference
    block_embed   - Block-wise PCA embedding
    joint_embed   - Joint VAE embedding  
    diffusion     - Diffusion model training
    generation    - Sample generation

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -s|--steps)
            STEPS="$2"
            shift 2
            ;;
        -f|--force-rerun)
            FORCE_RERUN=true
            shift
            ;;
        -e|--env)
            PYTHON_ENV="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to setup CUDA environment
setup_cuda_env() {
    print_status "Setting up CUDA environment..."
    
    # Comprehensive CUDA/TensorFlow warning suppression
    export TF_CPP_MIN_LOG_LEVEL=3  # Suppress all TF messages except errors
    export TF_ENABLE_ONEDNN_OPTS=0  # Disable oneDNN warnings
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}  # Set default GPU
    export CUDA_LAUNCH_BLOCKING=1  # Reduce CUDA warnings
    export CUDA_CACHE_DISABLE=1    # Disable CUDA cache warnings
    
    # Additional XLA/CUDA suppression
    export TF_XLA_FLAGS="--tf_xla_enable_xla_devices=false"
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=/dev/null"
    
    # Suppress Python warnings
    export PYTHONWARNINGS="ignore"
    
    # Try to load CUDA module if available (common on HPC systems)
    if command -v module &> /dev/null; then
        # Check if CUDA module is available
        if module avail cuda 2>/dev/null | grep -q cuda; then
            print_status "Loading CUDA module..."
            module load cuda 2>/dev/null || print_warning "Failed to load CUDA module"
        fi
    fi
    
    # Check if CUDA is available (quietly)
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    else
        print_warning "nvidia-smi not found. GPU may not be available."
    fi
}

# Function to activate Python environment
activate_env() {
    if [[ -n "$PYTHON_ENV" ]]; then
        print_status "Activating Python environment: $PYTHON_ENV"
        
        # Try conda first
        if command -v conda &> /dev/null; then
            eval "$(conda shell.bash hook)"
            conda activate "$PYTHON_ENV" 2>/dev/null || {
                print_warning "Failed to activate conda environment '$PYTHON_ENV'"
                # Try as virtualenv
                if [[ -f "$PYTHON_ENV/bin/activate" ]]; then
                    source "$PYTHON_ENV/bin/activate"
                    print_status "Activated virtualenv: $PYTHON_ENV"
                else
                    print_error "Could not find Python environment: $PYTHON_ENV"
                    exit 1
                fi
            }
        elif [[ -f "$PYTHON_ENV/bin/activate" ]]; then
            source "$PYTHON_ENV/bin/activate"
            print_status "Activated virtualenv: $PYTHON_ENV"
        else
            print_error "Could not find Python environment: $PYTHON_ENV"
            exit 1
        fi
    fi
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check if Python is available
    if ! command -v python &> /dev/null; then
        print_error "Python is not available in PATH"
        exit 1
    fi
    
    # Check if DiffuGene package is importable (suppress warnings during test)
    print_status "Testing DiffuGene import..."
    python -c "import DiffuGene; print('DiffuGene package imported successfully')" 2>/dev/null || {
        print_error "DiffuGene package import failed."
        print_error "This might be due to:"
        echo "  1. Missing installation: pip install -e ."
        echo "  2. CUDA library issues: try 'module load cuda' (on HPC systems)"
        echo "  3. Environment issues: check Python path and dependencies"
        echo ""
        echo "For debugging, try: python -c 'import DiffuGene'"
        exit 1
    }
    
    print_success "Dependencies check passed"
}

# Function to run the pipeline
run_pipeline() {
    print_status "Starting DiffuGene Pipeline"
    
    # Build command
    CMD="python -m DiffuGene.pipeline"
    
    if [[ -n "$CONFIG_FILE" ]]; then
        if [[ ! -f "$CONFIG_FILE" ]]; then
            print_error "Configuration file not found: $CONFIG_FILE"
            exit 1
        fi
        CMD="$CMD --config $CONFIG_FILE"
        print_status "Using config file: $CONFIG_FILE"
    fi
    
    if [[ -n "$STEPS" ]]; then
        # Convert comma-separated to space-separated
        STEPS_ARRAY=(${STEPS//,/ })
        CMD="$CMD --steps ${STEPS_ARRAY[@]}"
        print_status "Running steps: ${STEPS_ARRAY[@]}"
    fi
    
    if [[ "$FORCE_RERUN" == "true" ]]; then
        CMD="$CMD --force-rerun"
        print_status "Force rerun enabled"
    fi
    
    print_status "Executing: $CMD"
    echo ""
    
    # Run the pipeline
    eval $CMD
}

# Main execution
main() {
    print_status "DiffuGene Pipeline Runner Starting..."
    
    # Setup CUDA environment
    setup_cuda_env
    
    # Activate environment if specified
    activate_env
    
    # Check dependencies
    check_dependencies
    
    # Run pipeline
    run_pipeline
    
    print_success "Pipeline execution completed!"
}

# Trap to handle interrupts
trap 'print_error "Pipeline interrupted by user"; exit 130' INT

# Run main function
main "$@" 