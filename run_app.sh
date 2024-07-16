#!/bin/bash

# Default to development mode
MODE=${1:-dev}

# Common arguments for both modes
COMMON_ARGS="--model_name_or_path llava-hf/llava-v1.6-vicuna-13b-hf --use_hpu_graphs --bf16"

if [ "$MODE" = "dev" ]; then
    echo "Running in development mode..."
    python3 llavabridge.py $COMMON_ARGS
#elif [ "$MODE" = "prod" ]; then
#    echo "Running in production mode..."
#    gunicorn --bind 0.0.0.0:5000 "llavabridge:create_app()" $COMMON_ARGS
else
    echo "Invalid mode. Use 'dev' for development. 'prod' for production is not implemented yet."
    exit 1
fi
