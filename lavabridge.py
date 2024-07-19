import argparse
import logging
import time
from io import BytesIO
import PIL.Image
import requests
import torch
import base64
from flask import Flask, request, jsonify, Response
from PIL import Image
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BridgeTowerProcessor,
    BridgeTowerForImageAndTextRetrieval,
    pipeline
)
from optimum.habana.transformers.modeling_utils import adapt_transformers_to_gaudi

app = Flask(__name__)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Prometheus metrics
GENERATE_REQUESTS = Counter('generate_requests_total', 'Total number of generate requests')
PREDICT_REQUESTS = Counter('predict_requests_total', 'Total number of predict requests')
GENERATE_LATENCY = Histogram('generate_latency_seconds', 'Latency of generate requests')
PREDICT_LATENCY = Histogram('predict_latency_seconds', 'Latency of predict requests')

# Global variables to store model and configuration

processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")
model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-gaudi")

generator = None
generate_kwargs = None

def initialize_model(args):
    global generator, generate_kwargs

    adapt_transformers_to_gaudi()

    if args.bf16:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float32

    generator = pipeline(
        "image-to-text",
        model=args.model_name_or_path,
        torch_dtype=model_dtype,
        device="hpu",
    )

    generate_kwargs = {
        "lazy_mode": True,
        "hpu_graphs": args.use_hpu_graphs,
        "max_new_tokens": args.max_new_tokens,
        "ignore_eos": args.ignore_eos,
    }

    if args.use_hpu_graphs:
        from habana_frameworks.torch.hpu import wrap_in_hpu_graph
        generator.model = wrap_in_hpu_graph(generator.model)

@app.route('/generate', methods=['POST'])
def generate():
    GENERATE_REQUESTS.inc()
    with GENERATE_LATENCY.time():
        data = request.json
        image_input = data.get('image')
        prompt = data.get('prompt')

        if not image_input:
            return jsonify({"error": "Image input is required"}), 400

        try:
            image = get_image_from_input(image_input)
        except Exception as e:
            return jsonify({"error": f"Failed to load image: {str(e)}"}), 400

        start = time.perf_counter()
        result = generator([image], prompt=prompt, batch_size=1, generate_kwargs=generate_kwargs)
        end = time.perf_counter()
        duration = end - start

        # Extract text after "ASSISTANT:"
        full_text = result[0][0]['generated_text']
        assistant_text = full_text.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in full_text else full_text

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Explicit garbage collection
        import gc
        gc.collect()
        
        return jsonify({
            "text": assistant_text,
            "time_taken": duration * 1000,  # Convert to milliseconds
        })
    
    
@app.route('/predict', methods=['POST'])
def predict():
    PREDICT_REQUESTS.inc()
    with PREDICT_LATENCY.time():
        # Get the text and image input from the request
        data = request.json
        prompts = data.get('prompts')
        image_input = data.get('image')

        if not prompts or not image_input:
            return jsonify({"error": "Both prompts and image input are required"}), 400

        try:
            # Process the image
            image = get_image_from_input(image_input)

            results = []
            for prompt in prompts:
                # Perform prediction
                encoding = processor(image, prompt, return_tensors="pt")
                outputs = model(**encoding)
                confidence = outputs.logits[0,1].item()

                # Determine result based on confidence threshold
                result = "Yes" if confidence > 0.5 else "No"
                results.append({
                    "prompt": prompt,
                    "result": result,
                    "confidence": confidence
                })

            return jsonify({"predictions": results})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        
def get_image_from_input(image_input):
    if image_input.startswith('http://') or image_input.startswith('https://'):
        # It's a URL, fetch the image
        return Image.open(requests.get(image_input, stream=True, timeout=3000).raw)
    else:
        # Assume it's base64 encoded image data
        image_data = base64.b64decode(image_input)
        return Image.open(BytesIO(image_data))

def create_app():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="llava-hf/llava-v1.6-vicuna-13b-hf", type=str, help="Path to pre-trained model")
    parser.add_argument("--use_hpu_graphs", action="store_true", help="Whether to use HPU graphs or not.")
    parser.add_argument("--max_new_tokens", type=int, default=2000, help="Number of tokens to generate.")
    parser.add_argument("--bf16", action="store_true", help="Whether to perform generation in bf16 precision.")
    parser.add_argument("--ignore_eos", action="store_true", help="Whether to ignore eos, set False to disable it.")
    args = parser.parse_args()

    initialize_model(args)
    return app

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
    
