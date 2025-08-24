// script.js

document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // Stop the form from submitting normally
    
    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];
    const resultDiv = document.getElementById('results');
    const submitButton = document.getElementById('submit-button');
    const spinner = document.getElementById('loading-spinner');

    if (!file) {
        resultDiv.style.display = 'block';
        resultDiv.textContent = "Please select an image file first.";
        return;
    }

    // Show loading spinner, hide results, and disable button
    spinner.style.display = 'block';
    resultDiv.style.display = 'none';
    submitButton.disabled = true;

    const formData = new FormData();
    formData.append('image', file);

    try {
        // Send the image to the Python backend
        const response = await fetch('/evaluate-image', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();

        // Display the results from the backend
        resultDiv.style.display = 'block';
        resultDiv.textContent = result.message;

    } catch (error) {
        console.error('Error:', error);
        resultDiv.style.display = 'block';
        resultDiv.textContent = `An error occurred: ${error.message}. Make sure the Python server is running.`;
    } finally {
        // Hide loading spinner and re-enable button
        spinner.style.display = 'none';
        submitButton.disabled = false;
    }
});
