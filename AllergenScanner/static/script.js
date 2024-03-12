// script.js
const video = document.getElementById('video');
const captureButton = document.getElementById('capture');

// Access the user's camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
    } catch (error) {
        console.error('Error accessing camera:', error);
    }
}

// Capture an image from the camera
function takePicture() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert the image to base64 data URL
    const imageData = canvas.toDataURL('image/jpeg');

    // Send the image data to your Python script (via an API endpoint)
    fetch('/api/predict-allergens', { method: 'POST', body: imageData });
}

// Initialize camera on page load
window.addEventListener('DOMContentLoaded', () => {
    startCamera();
    captureButton.addEventListener('click', takePicture);
});
