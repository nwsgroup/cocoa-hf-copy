#!/bin/bash

# Exit on error
set -e

# Function to read JSON values using jq
get_json_value() {
    local key=$1
    jq -r "$key" models.json
}

# Function to print usage information
usage() {
    echo "Usage: $0 -m MODEL_TYPE [-d DATASET] [-o OUTPUT_DIR] [-b BATCH_SIZE] [-e EPOCHS] [-l LR]"
    echo
    echo "Options:"
    echo "  -m MODEL_TYPE     Model type (use --list to see available models)"
    echo "  -d DATASET        Dataset name (default: from config)"
    echo "  -o OUTPUT_DIR     Output directory (default: ./output/outputs_\${MODEL_TYPE})"
    echo "  -b BATCH_SIZE     Batch size (default: from config)"
    echo "  -e EPOCHS         Number of epochs (default: from config)"
    echo "  -l LR            Learning rate (default: from config)"
    echo "  -h, --help       Show this help message"
    echo "  --list           List available models"
    exit 1
}

# Function to list available models
list_available_models() {
    echo "Available models:"
    jq -r '.models | keys[]' models.json | while read -r model; do
        description=$(jq -r ".models.${model}.description" models.json)
        echo "  - $model: $description"
    done
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed. Please install jq first."
    exit 1
fi

# Check if config file exists
if [ ! -f "models.json" ]; then
    echo "Error: models.json configuration file not found"
    exit 1
fi

# Handle --list and --help before other options
if [[ "$1" == "--list" ]]; then
    list_available_models
    exit 0
fi

if [[ "$1" == "--help" ]]; then
    usage
    exit 0
fi

# Parse command line arguments
while getopts "m:d:o:b:e:l:h" opt; do
    case $opt in
        m) MODEL_TYPE="$OPTARG" ;;
        d) DATASET="$OPTARG" ;;
        o) OUTPUT_DIR="$OPTARG" ;;
        b) BATCH_SIZE="$OPTARG" ;;
        e) EPOCHS="$OPTARG" ;;
        l) LEARNING_RATE="$OPTARG" ;;
        h) usage ;;
        \?) echo "Invalid option -$OPTARG" >&2; usage ;;
    esac
done

# Validate MODEL_TYPE
if [ -z "$MODEL_TYPE" ]; then
    echo "Error: Model type (-m) is required"
    echo "Use --list to see available models"
    usage
fi

# Check if model exists in config
if ! jq -e ".models.$MODEL_TYPE" models.json > /dev/null; then
    echo "Error: Invalid model type: $MODEL_TYPE"
    list_available_models
    exit 1
fi

# Get model configuration
MODEL_NAME=$(get_json_value ".models.$MODEL_TYPE.name")
DEFAULT_BATCH_SIZE=$(get_json_value ".models.$MODEL_TYPE.batch_size")
DEFAULT_LR=$(get_json_value ".models.$MODEL_TYPE.learning_rate")
DEFAULT_DATASET=$(get_json_value ".default_settings.dataset")
DEFAULT_EPOCHS=$(get_json_value ".default_settings.epochs")

# Set variables with defaults from config
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/outputs_${MODEL_TYPE}}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
LEARNING_RATE="${LEARNING_RATE:-$DEFAULT_LR}"
DATASET="${DATASET:-$DEFAULT_DATASET}"
EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
RUN_NAME="${MODEL_TYPE}-cocoa-run"
MODEL_ID="${MODEL_TYPE}-cocoa"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Configure Weights & Biases
export WANDB_PROJECT="cocoa-image-classification"
export WANDB_RUN_NAME="${MODEL_TYPE}"

# Print configuration
echo "Starting training with the following configuration:"
echo "Model Type: $MODEL_TYPE"
echo "Model Name: $MODEL_NAME"
echo "Dataset: $DATASET"
echo "Output Directory: $OUTPUT_DIR"
echo "Batch Size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LEARNING_RATE"
echo "Run Name: $RUN_NAME"

# Execute training
python main.py \
    --dataset_name "$DATASET" \
    --output_dir "$OUTPUT_DIR" \
    --with_tracking \
    --report_to wandb \
    --label_column_name label \
    --ignore_mismatched_sizes \
    --do_eval \
    --model_name_or_path "$MODEL_NAME" \
    --learning_rate "$LEARNING_RATE" \
    --lr_scheduler_type "cosine" \
    --weight_decay 0.01 \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --logging_strategy epoch \
    --eval_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end "true" \
    --save_total_limit "$(get_json_value '.default_settings.save_total_limit')" \
    --seed "$(get_json_value '.default_settings.seed')" \

#--num_warmup_steps 4 \
#--push_to_hub \
#--push_to_hub_model_id "$MODEL_ID" \