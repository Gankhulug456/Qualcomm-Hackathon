import React, { useState } from "react";
import axios from "axios";

const FileUploader = () => {
  const [file, setFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert("Please select a file to upload!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      setLoading(true);
      setError(null);
      const response = await axios.post("http://127.0.0.1:8000/analyze", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setAnalysisResult(response.data);
    } catch (err) {
      setError("An error occurred while analyzing the document.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="file-uploader">
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".doc, .docx, .pdf" onChange={handleFileChange} />
        <button type="submit">Upload & Analyze</button>
      </form>
      {loading && <p>Analyzing document...</p>}
      {error && <p className="error">{error}</p>}
      {analysisResult && (
        <div className="analysis-result">
          <h3>Analysis Result</h3>
          <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default FileUploader;
