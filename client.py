import cv2
import numpy as np
import streamlit as st
import requests
import base64
import time
import argparse
import pandas as pd

# change VideoCapture(0) to different camera number if you have multiple cameras
# Check your camera number with:
# sudo apt install v4l-utils
# v4l2-ctl --list-devices

if 'cap' not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

def capture_frame():
    while(True):
        ret, frame = st.session_state.cap.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(ret)
    return np.zeros((480, 640, 3), dtype=np.uint8)

def encode_frame(frame, quality=90):
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return base64.b64encode(buffer).decode('utf-8')

def send_request(endpoint, image, prompt=None):
    if endpoint == "metrics":
        response = requests.get(f"http://{st.session_state.ip}:{st.session_state.port}/metrics")
        response.raise_for_status()
        if 'application/json' in response.headers.get('Content-Type', ''):
            metrics = response.json()
            return metrics
        else:
            # If not JSON, print the raw text content
            return response.text
    encoded_image = encode_frame(image)
    data = {"image": encoded_image}
    if prompt:
        if endpoint == "predict":
            data["prompts"] = prompt
        else:
            data["prompt"] = prompt
    response = requests.post(f"http://{st.session_state.ip}:{st.session_state.port}/{endpoint}", json=data)
    return response.json()

def parse_metrics(metrics_text):
    """Parse the metrics text into a structured format."""
    lines = metrics_text.split('\n')
    data = []
    for line in lines:
        if line and not line.startswith('#'):
            # Split the line into name and value
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                value = float(parts[-1])
                rounded_value = np.round(value, decimals=3)
                data.append({"Metric": name, "Value": rounded_value})
    return pd.DataFrame(data)

def display_metrics(df):
    """Display the metrics in a readable format using Streamlit."""
    st.title("Metrics Dashboard")
    
    # Display as a table
    st.subheader("Metrics Table")
    st.dataframe(df.style.format({'Value': '{:,.3f}'}), width=800)
    
    # Display as a bar chart
    st.subheader("Metrics Visualization")
    st.bar_chart(df.set_index("Metric"))

def main():
    parser = argparse.ArgumentParser(description='Client for bjornservices_llavabridge')
    parser.add_argument('--ip', default='192.55.42.78', help='IP address of the server')
    parser.add_argument('--port', default=5000, type=int, help='Port number of the server')
    args = parser.parse_args()
    st.session_state.ip = args.ip
    st.session_state.port = args.port

    st.set_page_config(page_title="Dashboard")
    st.title("Webcam Capture, BridgeTower and LLaVA Interface")

    frame = capture_frame()

    # Create a placeholder for the webcam feed
    frame_placeholder = st.empty()

    # Text area for generate prompt
    default_generate_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to questions. USER: <image>\nWhat is in the image? ASSISTANT:"
    generate_prompt = st.text_area("Enter prompt for generation:", value=default_generate_prompt, height=100)

    # Text area for predict prompts
    default_predict_prompts = "Is there a person in this image?\nIs there a cat in this image?\nIs there a car in this image?"
    predict_prompts = st.text_area("Enter prompts for prediction (one per line):", value=default_predict_prompts, height=100)

    # Create columns for buttons
    col1, col2, col3 = st.columns(3)

    # Generate button
    if col1.button("Generate"):
        with st.spinner("Generating..."):
            frame = capture_frame()
            frame_placeholder.image(frame, channels="RGB")
            response = send_request("generate", frame, generate_prompt)
            st.write(f"Generated text: {response.get('text', 'No text generated')}")
            st.write(f"Time taken: {response.get('time_taken', 'N/A')} ms")

    # Predict button
    if col2.button("Predict"):
        with st.spinner("Predicting..."):
            frame = capture_frame()
            frame_placeholder.image(frame, channels="RGB")
            prompts = [prompt.strip() for prompt in predict_prompts.split('\n') if prompt.strip()]
            response = send_request("predict", frame, prompts)
            predictions = response.get('predictions', [])
            for pred in predictions:
                st.write(f"Prompt: {pred['prompt']}")
                st.write(f"Prediction result: {pred['result']}")
                st.write(f"Confidence: {pred['confidence']}")
                st.write("---")

    # Metric button
    if col3.button("Show Metrics"):
        with st.spinner("Fetching Metrics..."):
            # Assuming send_request can handle a "metric" action and returns metrics data
            metrics_text = send_request("metrics", None, None) 
            if metrics_text:
                df = parse_metrics(metrics_text)
                display_metrics(df) # Assuming no frame or prompt is needed for metrics

    # Continuously update the webcam feed
    while True:
        frame = capture_frame()
        frame_placeholder.image(frame, channels="RGB")
        time.sleep(1)  # Wait for 1 second before updating again

if __name__ == "__main__":
    main()
