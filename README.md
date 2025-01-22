# Legal Document Analyzer

## Authors
- Arjun Dasgupta
- Gankhulug Bayaraa
- Golam Raiyan
- Zoe Dauphinee

## Description
A secure, offline tool that automates legal document analysis, including contract review, case simulation, and intellectual property management. Designed to enhance efficiency and ensure data privacy, it reduces manual workload for legal professionals.

## Setup Instructions

### Requirements
-LM Studio
- Python 3.11+
- Install required dependencies using pip
  -In requirements.txt
  
## Running the Application
1. In the terminal, run the app using:
   uvicorn api:app --reload
2. Choose an LLM model on LM Studio and start the server with port "1234".
3. Access the app by visiting the local server address generated on the terminal.
4. Upload a legal document (DOCX, PDF, or TXT format) through the UI and verify the analysis and summary.

### Analyze a Document
- Visit the app's UI at the local server address generated and upload a document for analysis.
- The document will be broken down into clauses and each clause will be evaluated for risk level (Low/Medium/High).

### API Endpoints
- **GET `/diagnostics`**: Check the app's status and configurations.
- **POST `/analyze`**: Upload a document and analyze it for risks.
- **POST `/api/prompt`**: Trigger specific actions (e.g., contract analysis or document summarization).

## Notes
- Ensure that your environment variables are correctly set for the LM Studio integration.
- This app relies on FastAPI and supports asynchronous operations for document analysis.
- If you encounter issues with unsupported file types, ensure the uploaded document is in a supported format (PDF, DOCX, or TXT).


Presentation link: https://docs.google.com/presentation/d/1MIRv9SQfyYNrP0dFFoxHBYRPrNO-ZC0NUKjnXQfMEeo/edit?usp=sharing
