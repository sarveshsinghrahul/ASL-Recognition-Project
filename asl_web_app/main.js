/*
 * === THE REAL, FINAL, VISUAL FIX ===
 * This version adds a 'fixLandmarkCoordinates' function.
 * This new function corrects the "slight offset" by
 * un-stretching and un-offsetting the coordinates
 * before they are drawn.
 *
 * THE ACCURACY LOGIC IS UNCHANGED.
 * THE DATA FLIP LOGIC IS UNCHANGED.
*/

// --- IMPORTANT! ---
// You must update this array to match the classes your model was trained on.
const CLASSES = [
  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
  'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
  'del', 'nothing', 'space'
];

// A 1D vector of 63 zeros. This is our "nothing" feature.
const ZERO_VECTOR = Array(63).fill(0.0);

// Get references to HTML elements
const videoElement = document.getElementById("input_video");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const predictionText = document.getElementById("prediction");
const liveView = document.getElementById("liveView");

// Get the hidden canvas for pre-processing
const hiddenCanvas = document.getElementById('hidden_canvas');
const hiddenCtx = hiddenCanvas.getContext('2d');

// Global model variable
let customModel;
let hands; // Make 'hands' global so the loop can see it


// --- *** NEW FUNCTION *** ---
/**
 * This function fixes the visual offset.
 * It converts the coordinates from the 480x480 cropped space
 * to the 640x480 full video space.
 */
function fixLandmarkCoordinates(landmarks, videoWidth, videoHeight) {
  const squareSize = videoHeight; // 480
  const offset = (videoWidth - squareSize) / 2; // (640 - 480) / 2 = 80
  
  const fixedLandmarks = [];
  
  for (const landmark of landmarks) {
    const fixedX = (landmark.x * squareSize + offset) / videoWidth;
    const fixedY = landmark.y; // Y is already correct (0-1.0 relative to 480)
    
    // We don't need to fix Z, but we must pass it along
    fixedLandmarks.push({
      x: fixedX,
      y: fixedY,
      z: landmark.z
    });
  }
  
  return fixedLandmarks;
}
// --- *** END NEW FUNCTION *** ---


/**
 * The main prediction logic, now inside a callback function
 */
function onHandResults(results) {
  // Clear the *visible* canvas
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

  let flatLandmarks;

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = results.multiHandLandmarks[0];
    
    // --- THIS IS THE VISUAL FIX ---
    // We fix the coordinates *before* drawing them.
    const fixedVisualLandmarks = fixLandmarkCoordinates(
      landmarks, 
      videoElement.videoWidth, 
      videoElement.videoHeight
    );
    // --- END FIX ---
    
    // Draw the *fixed* landmarks on the *visible* canvas
    window.drawLandmarks(canvasCtx, fixedVisualLandmarks, { color: "#FF0000", radius: 5 });
    window.drawConnectors(canvasCtx, fixedVisualLandmarks, window.HAND_CONNECTIONS, { color: "#00FF00" });

    // Pre-process the *original* landmarks for the model
    flatLandmarks = preprocessLandmarks(landmarks);

  } else {
    // No hand was detected. Feed the ZERO_VECTOR.
    flatLandmarks = ZERO_VECTOR;
  }
  
  // Run prediction (this is unchanged and still accurate)
  const prediction = runModel(flatLandmarks);
  
  // Display the result
  const predictedClassIndex = prediction.index;
  const predictedClassName = CLASSES[predictedClassIndex];
  predictionText.innerText = predictedClassName;
}

/**
 * Pre-processes the landmark data (for the model).
 * This is UNCHANGED and correct.
 */
function preprocessLandmarks(landmarks) {
  const basePoint = landmarks[0]; // Wrist
  const processedLandmarks = [];
  
  for (const landmark of landmarks) {
    // Subtract the base (wrist) coordinates for x, y, and z
    const relativeX = landmark.x - basePoint.x;
    const relativeY = landmark.y - basePoint.y;
    const relativeZ = landmark.z - basePoint.z;
    
    processedLandmarks.push(relativeX, relativeY, relativeZ);
  }
  return processedLandmarks;
}

/**
 * Runs inference with the custom TensorFlow.js model
 * This is UNCHANGED and correct.
 */
function runModel(flatLandmarks) {
  return window.tf.tidy(() => {
    const inputTensor = window.tf.tensor2d([flatLandmarks], [1, 63]);
    const outputTensor = customModel.predict(inputTensor);
    const probabilities = outputTensor.dataSync();
    const maxProbabilityIndex = outputTensor.argMax(1).dataSync()[0];

    return {
      index: maxProbabilityIndex,
      probability: probabilities[maxProbabilityIndex]
    };
  });
}

/**
 * Creates our own prediction loop
 * This is UNCHANGED and correct.
 */
async function predictLoop() {
  // 1. Calculate the source (video) and destination (hidden canvas) rectangles
  const videoWidth = videoElement.videoWidth;
  const videoHeight = videoElement.videoHeight;
  const squareSize = videoHeight; // Use the smaller dimension (480)
  const srcX = (videoWidth - squareSize) / 2; // (640 - 480) / 2 = 80
  const srcY = 0;

  // --- THIS IS THE ACCURACY FIX ---
  // 2. Flip the hidden canvas context horizontally
  hiddenCtx.save();
  hiddenCtx.scale(-1, 1);
  hiddenCtx.translate(-hiddenCanvas.width, 0);
  // --- END ACCURACY FIX ---
  
  // 3. Draw the center-cropped video frame onto the *now flipped* HIDDEN canvas
  hiddenCtx.drawImage(
    videoElement, // source
    srcX,         // source x
    srcY,         // source y
    squareSize,   // source width (480)
    squareSize,   // source height (480)
    0,            // dest x
    0,            // dest y
    hiddenCanvas.width,  // dest width (480)
    hiddenCanvas.height  // dest height (480)
  );
  
  // 4. Un-flip the context so it's ready for the next frame
  hiddenCtx.restore();
  
  // 5. Send the *flipped, cropped* HIDDEN CANVAS to MediaPipe
  // This will trigger the 'onHandResults' callback
  await hands.send({ image: hiddenCanvas });
  
  // Request the next frame to continue the loop
  requestAnimationFrame(predictLoop);
}

/**
 * Main function to orchestrate the application startup
 * This is UNCHANGED and correct.
 */
async function runApp() {
  predictionText.innerText = "Loading model...";
  // Load the custom model
  customModel = await window.tf.loadGraphModel("./web_model/model.json");
  console.log("Custom ASL model loaded.");
  predictionText.innerText = "Loading hand detector...";

  // --- Configure the OLD MediaPipe Hands ---
  hands = new window.Hands({
    locateFile: (file) => {
      // This function strips the internal path
      const filename = file.split('/').pop();
      return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${filename}`;
    }
  });

  // --- THESE ARE THE 1:1 PYTHON SETTINGS ---
  hands.setOptions({
    staticImageMode: true, // This is the key
    modelComplexity: 1,    // This loads the 'full' model
    maxNumHands: 1,
    minDetectionConfidence: 0.5 // Matches your Python script
  });
  // --- END SETTINGS ---

  // Set the callback function
  hands.onResults(onHandResults);

  // --- Manually get the camera stream ---
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
      }
    });
    videoElement.srcObject = stream;
  } catch (err) {
    console.error("Error accessing webcam:", err);
    predictionText.innerText = "Error: Webcam failed.";
    return;
  }

  // Use 'onplaying' to reliably start the loop
  videoElement.onplaying = () => {
    const videoWidth = videoElement.videoWidth;
    const videoHeight = videoElement.videoHeight;

    // Set VISIBLE canvas size
    canvasElement.width = videoWidth;
    canvasElement.height = videoHeight;
    liveView.style.width = videoWidth + "px";
    liveView.style.height = videoHeight + "px";
    
    // Set HIDDEN canvas size (SQUARE)
    const squareSize = videoHeight; // e.g., 480
    hiddenCanvas.width = squareSize;
    hiddenCanvas.height = squareSize;
    
    // Start the prediction loop!
    predictLoop();
    
    predictionText.innerText = "Ready!";
    console.log("MediaPipe Hands (old library) loaded in STATIC mode with 1:1 cropped input.");
  };
}

// Start the application
runApp();