import React from "react";
import FileUploader from "./components/FileUploader";

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>Legal Document Analyzer</h1>
      </header>
      <main>
        <FileUploader />
      </main>
    </div>
  );
}

export default App;
