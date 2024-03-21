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
    fetch('/api/process-image', { 
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageData }) })
        .then(response => response.json()) // Parse the JSON response
        .then(data => {
            console.log(data)
            document.getElementById('prediction').textContent = `Prediction ${data.prediction}`;
            /* document.getElementById('eggs').textContent = `Eggs: ${data.eggs}`;
            document.getElementById('milk').textContent = `Milk: ${data.milk}`;
            document.getElementById('nuts').textContent = `Nuts: ${data.nuts}`;
            document.getElementById('peanuts').textContent = `Peanuts: ${data.peanuts}`;
            document.getElementById('soja').textContent = `Soja: ${data.soja}`;
            document.getElementById('molluscs').textContent = `Molluscs: ${data.molluscs}`;
            document.getElementById('fish').textContent = `Fish: ${data.fish}`;
            document.getElementById('lupin').textContent = `Lupin: ${data.lupin}`;
            document.getElementById('crustaceans').textContent = `crustaceans: ${data.crustaceans}`;
            document.getElementById('sesame').textContent = `Sesame: ${data.sesame}`;
            document.getElementById('mustard').textContent = `Mustard: ${data.mustard}`;
            document.getElementById('celery').textContent = `Celery: ${data.celery}`;
            document.getElementById('sulphites').textContent = `Sulphites: ${data.sulphites}`; */
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
