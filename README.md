# Real-Time ASL Alphabet Recognition

This project uses a custom-trained Convolutional Neural Network (CNN) with TensorFlow to recognize American Sign Language (ASL) alphabet hand patterns in real-time from a webcam.

To solve Python dependency conflicts, this project uses a two-process architecture:
* **Process A (Sender):** Runs in a separate Conda environment to handle hand-finding with MediaPipe.
* **Process B (Receiver):** Runs in the main TensorFlow GPU environment, receives cropped images, and performs the ASL classification.



[Image of the ASL alphabet]


## 🛠️ Tech Stack
* **Model:** TensorFlow (Keras)
* **Hand Detection:** MediaPipe (via `env_mediapipe`)
* **Real-Time Video:** OpenCV
* **Inter-Process Communication:** ZeroMQ (pyzmq)
* **Environment:** Conda (running in WSL 2)

---

## ⚙️ Setup & Installation

You must create two separate and isolated Conda environments.

### 1. Environment A (The Hand-Finder)
This environment will run `sender_mediapipe.py`.

```bash
# Create the env
conda create --name env_mediapipe python=3.10

# Activate it
conda activate env_mediapipe

# Install old, specific packages
pip install mediapipe==0.10.11 opencv-python pyzmq