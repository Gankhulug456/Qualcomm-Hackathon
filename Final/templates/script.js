document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('file').addEventListener('change', function () {
        var fileName = this.files[0] ? this.files[0].name : 'No file chosen';
        document.getElementById('file-name').textContent = fileName;

        // Show loading indicator while the file is being analyzed
        document.getElementById('loading-indicator').style.display = 'block';

        // Create FormData object to send the file
        var formData = new FormData();
        formData.append('file', this.files[0]);

        // Use fetch to send the file to the backend
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())  // Expect HTML response with analysis result
        .then(data => {
            // Hide loading indicator
            document.getElementById('loading-indicator').style.display = 'none';
            
            // Display the result
            document.getElementById('result').innerHTML = data;
        })
        .catch(error => {
            // Hide loading indicator and show error if something goes wrong
            document.getElementById('loading-indicator').style.display = 'none';
            console.error('Error:', error);
            alert('An error occurred while analyzing the document.');
        });
    });
});
