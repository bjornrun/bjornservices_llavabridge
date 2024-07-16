# Bjorn Services an AI Microservices Suite: LLaVA BridgeTower component

This project consists of two main components: a server part (LLaVA bridge) running on Gaudi 2, and a client part running on a laptop with a webcam.

## Server: LLaVA Bridge

The LLaVA bridge is the server component of this project. It runs on Gaudi 2 hardware and is responsible for processing requests from the client. This component utilizes the LLaVA (Large Language and Vision Assistant) model to analyze images and generate responses.

Key features:
- Runs on Gaudi 2 hardware for high-performance computing
- Processes image analysis requests
- Utilizes the LLaVA and BridgeTower models for advanced image understanding and response generation

## Client: Webcam Interface

The client component runs on a laptop equipped with a webcam. It captures images from the webcam and sends them to the server for processing.

Key features:
- Runs on a laptop with a webcam
- Captures frames from the webcam
- Encodes and sends image data to the server
- Displays responses received from the server

## How It Works

1. The client captures a frame from the webcam.
2. The image is encoded and sent to the LLaVA bridge server.
3. The server processes the image using the LLaVA model.
4. The server sends back a response based on the image analysis.
5. The client displays the response to the user.

## How to run the client


Follow these steps to set up the project and run the Streamlit app.

### Step 1: Create a Virtual Environment

Create a virtual environment named `venv`.

```bash
python3 -m venv venv
```

### Step 2: Activate the Virtual Environment
Activate the virtual environment. Use the following command for Unix-based systems (Linux and macOS):

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies
Install the required dependencies listed in required_client.txt.

```bash
pip install -r requirements_client.txt
```

### Step 4: Run the Streamlit App
Run the Streamlit app using the following command:

```bash
streamlit run client.py
```
