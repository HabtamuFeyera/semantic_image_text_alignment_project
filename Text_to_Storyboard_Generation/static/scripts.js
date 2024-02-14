function generateStoryboard() {
    var adText = document.getElementById("adText").value;
    var loadingIndicator = document.getElementById("loadingIndicator");
    var resultDiv = document.getElementById("result");

    // Show loading indicator
    loadingIndicator.style.display = "block";

    // Send ad description to server for storyboard generation
    fetch('/generate_storyboard', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: adText })
    })
    .then(response => response.json())
    .then(data => {
        // Hide loading indicator
        loadingIndicator.style.display = "none";

        // Display generated storyboard
        resultDiv.innerHTML = `<img src="${data.generated_storyboard}" alt="Generated Storyboard">`;
    })
    .catch(error => {
        console.error('Error:', error);
        // Hide loading indicator
        loadingIndicator.style.display = "none";
        // Display error message
        resultDiv.innerHTML = `<p>Error generating storyboard. Please try again later.</p>`;
    });
}
