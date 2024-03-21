// script.js
const video = document.getElementById('video');
const captureButton = document.getElementById('captureButton');

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

    video.style.display = 'none'; // Hide the video element
    captureButton.disabled = true;

    // Show a loading spinner while processing the image
    const loadingSpinner = document.createElement('div');
    loadingSpinner.classList.add('loader'); 
    document.body.appendChild(loadingSpinner);

    // Convert the image to base64 data URL
    const imageData = canvas.toDataURL('image/jpeg');

    // Send the image data to your Python script (via an API endpoint)
    fetch('/api/process-image', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData }) })
        .then(response => response.json()) // Parse the JSON response
        .then(data => {
            console.log(data)
            window.location.href = `/prediction.html?data=${encodeURIComponent(JSON.stringify(data))}`;
        })
        .catch(error => {
            console.error('Error processing image.', error);
        });
}

// Initialize camera on page load
window.addEventListener('DOMContentLoaded', () => {
    startCamera();
    captureButton.addEventListener('click', takePicture);
});
