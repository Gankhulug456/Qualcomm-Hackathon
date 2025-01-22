document.addEventListener("DOMContentLoaded", function () {
  const fileUploadContainer = document.querySelector(".file-upload-container");
  const fileInput = document.getElementById("file");
  const fileNameLabel = document.getElementById("file-name");

  fileUploadContainer.addEventListener("click", function () {
    fileInput.click();
  });

  fileInput.addEventListener("change", function () {
    var fileName = this.files[0] ? this.files[0].name : "No file chosen";
    fileNameLabel.textContent = fileName;
    const processingText = document.getElementById("processing-text");
    const resultBox = document.getElementById("result-box");
    const inputElement = document.getElementById("input");

    processingText.style.display = "block";
    resultBox.style.display = "none";
    inputElement.style.display = "none";
    fileUploadContainer.classList.add("processing");
    processingText.style.opacity = 0;
    setTimeout(() => {
      processingText.style.transition = "opacity 0.5s";
      processingText.style.opacity = 1;
    }, 10);

    document.querySelector(".greeting").style.display = "none";
    document.querySelector(".prompt-boxes").style.display = "none";

    var formData = new FormData();
    formData.append("file", this.files[0]);

    fetch("/analyze", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.text())
      .then((data) => {
        processingText.style.transition = "opacity 1s";
        processingText.style.opacity = 0;
        setTimeout(() => {
          resultBox.style.display = "block";
          resultBox.style.opacity = 0;
          resultBox.innerHTML = data;
          setTimeout(() => {
            resultBox.style.transition = "opacity 1s";
            resultBox.style.opacity = 1;
          }, 10);
        }, 500);
        fileUploadContainer.classList.remove("processing");
      })
      .catch((error) => {
        processingText.style.display = "none";
        fileUploadContainer.classList.remove("processing");
        console.error("Error:", error);
        alert("An error occurred while analyzing the document.");
      });
  });
});
