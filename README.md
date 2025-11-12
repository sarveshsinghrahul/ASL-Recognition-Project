# Real-Time ASL Alphabet Recognition

This project uses a machine learning model to recognize American Sign Language (ASL) alphabet signs in real-time from a webcam.

The final, working solution uses a **Landmark-Based MLP** (Multilayer Perceptron), which is a tiny, fast model trained on hand skeleton data. This approach was chosen after initial models, trained on raw pixels (CNNs), proved to be inaccurate in real-world webcam feeds.

---

## 🚀 Final Solution: Landmark-Based MLP

This is the final, working model. It is highly accurate because it is immune to different backgrounds and lighting conditions.

* **Model:** `asl_landmark_model.keras`
* **Notebook:** `train_landmarks.ipynb`
* **Prediction Script:** `predict.py`

### How It Works
1.  **Data Creation:** A script (`create_landmark_dataset_v2.py`) runs MediaPipe on all 87,000+ Kaggle images to extract the 21 hand landmarks (the "skeleton"). This converts each image into a 63-point vector (21 * 3 coordinates).
2.  **Training:** A simple, dense (MLP) model is trained on this new dataset of vectors. This model learns the *geometry* of the hand's pose, not the pixels.
3.  **Real-Time Prediction:** The live script (`predict.py`) does the same thing:
    * MediaPipe finds the hand landmarks in the webcam feed.
    * These 63 landmark numbers are fed to the tiny landmark model.
    * The model makes an instant prediction.

### Setup & Run
This project requires a single, stable, CPU-only environment that can run both TensorFlow and MediaPipe.

1.  **Create the Environment:**
    ```bash
    # Create a new, clean env with Python 3.10
    conda create --name asl_cpu_env python=3.10 -y
    conda activate asl_cpu_env
    ```
2.  **Install the Exact Compatible Libraries:**
    ```bash
    # Install the old (but compatible) TF and MediaPipe
    pip install tensorflow mediapipe
    
    # Install the rest of the tools
    pip install opencv-python matplotlib pandas scikit-learn ipykernel
    ```
3.  **Run the App:**
    ```bash
    python predict.py
    ```

---

## 💡 Project Journey & Model Evolution

This final solution was reached after two previous models failed to generalize to a real-world webcam.

### Model 1: The "Book Smart" CNN (Overfit)
* **Model File:** `model_v1_overfit.keras` (Your original model)
* **What it was:** A custom CNN trained on the raw `(200, 200)` Kaggle images with no augmentation.
* **Result:** `val_accuracy: 99.7%`. It perfectly "memorized" the training and validation images.
* **The Lesson:** This model failed *miserably* on test images and the webcam. It was **overfit** and had learned a "crutch": it associated the hand signs with the **perfect black background** of the Kaggle dataset.

### Model 2: The "Tougher" CNN (Transfer Learning)
* **Model File:** `model_v2_mobilenet.keras` (Your `model1.keras`)
* **What it was:** A `MobileNetV2` (Transfer Learning) model trained with aggressive data augmentation (zoom, rotation, brightness/contrast).
* **Result:** Much better than Model 1, but still "way off" and unstable.
* **The Lesson:** This proved the core problem was **domain mismatch**. Even with augmentation, the model was still being distracted by the "messy" real-world background. The pixel data itself was the problem.

### Model 3: The "Landmark Fix" (The Final Solution)
* **Model File:** `asl_landmark_model.keras`
* **What it was:** A complete change in strategy. Instead of training on *pixels*, I trained a simple MLP model on the *3D landmark coordinates* of the hand's skeleton.
* **Result:** This model is **immune to backgrounds and lighting** because it *only* sees the hand's pose data. It is fast, lightweight, and accurately generalizes to a real-world webcam feed.