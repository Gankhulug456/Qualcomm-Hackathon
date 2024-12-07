from openai import OpenAI
from docx import Document
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import json
import mimetypes
from typing import List, Dict, Any, Union

# Load environment variables
load_dotenv()

# Initialize FastAPI app and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")


class DocumentAnalyzer:
    def __init__(self) -> None:
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("API_KEY", "lm-studio")
        )

    def read_docx(self, file_path: str) -> str:
        """Read content from a DOCX file."""
        try:
            doc = Document(file_path)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {e}")

    def read_pdf(self, file_path: str) -> str:
        """Read content from a PDF file."""
        try:
            reader = PdfReader(file_path)
            return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")

    def read_txt(self, file_path: str) -> str:
        """Read content from a TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading TXT file: {e}")

    def read_file(self, file_path: str) -> str:
        """Determine file type and extract text."""
        mime_type, _ = mimetypes.guess_type(file_path)
        try:
            if mime_type == "application/pdf":
                return self.read_pdf(file_path)
            elif mime_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
            ]:
                return self.read_docx(file_path)
            elif mime_type and mime_type.startswith("text/"):
                return self.read_txt(file_path)
            else:
                raise ValueError("Unsupported file type")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    def extract_clauses(self, document_text: str) -> List[str]:
        """Break document into clauses using AI."""
        prompt = f"""
        Break the following contract into separate clauses:
        {document_text}
        Return only the separated clauses.
        """
        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_ID", "model-identifier"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.split('\n\n')

    def analyze_clause(self, clause_text: str) -> Dict[str, Union[str, Any]]:
        """Analyze a single clause for risks."""
        prompt = f"""
        Analyze the following contract clause for risk level and provide:
        1. Risk level (Low, Medium, High)
        2. Brief explanation of the risk
        
        Clause: {clause_text}
        """
        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_ID", "model-identifier"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        analysis = response.choices[0].message.content
        return {
            "clause_text": clause_text,
            "analysis": analysis,
            "risk_level": self._extract_risk_level(analysis)
        }

    def _extract_risk_level(self, analysis_text: str) -> str:
        """Extract risk level from analysis text."""
        lower_text = analysis_text.lower()
        if "high risk" in lower_text or "high" in lower_text:
            return "High"
        elif "medium risk" in lower_text or "medium" in lower_text:
            return "Medium"
        return "Low"

    def calculate_risk_score(self, analyzed_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate an overall risk score for the document."""
        if not analyzed_clauses:
            return {"overall_score": 0, "risk_breakdown": {}}

        clause_weights = {
            "Indemnification": 1.5,
            "Termination": 1.3,
            "Confidentiality": 1.2,
            "Default": 1.0,
        }
        risk_weights = {"High": 3, "Medium": 2, "Low": 1}
        total_weighted_score = 0
        total_weight = 0
        risk_breakdown = {"High": 0, "Medium": 0, "Low": 0}

        for clause in analyzed_clauses:
            risk_level = clause["risk_level"]
            risk_score = risk_weights.get(risk_level, 1)
            importance_weight = next(
                (weight for key, weight in clause_weights.items() if key.lower() in clause["clause_text"].lower()),
                1.0
            )
            total_weighted_score += risk_score * importance_weight
            total_weight += importance_weight
            risk_breakdown[risk_level] += 1

        overall_score = (total_weighted_score / (total_weight * 3)) * 100
        return {
            "overall_score": round(overall_score, 2),
            "risk_breakdown": risk_breakdown,
        }

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Main method to analyze a document."""
        document_text = self.read_file(file_path)
        clauses = self.extract_clauses(document_text)
        analyzed_clauses = [self.analyze_clause(clause) for clause in clauses]
        risk_data = self.calculate_risk_score(analyzed_clauses)
        return {
            "filename": os.path.basename(file_path),
            "clauses": analyzed_clauses,
            "overall_score": risk_data["overall_score"],
            "risk_breakdown": risk_data["risk_breakdown"],
        }


@app.get("/", response_class=HTMLResponse)
async def get_ui() -> HTMLResponse:
    """Render UI for inputting file path."""
    return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head><title>Document Analyzer</title></head>
        <body>
            <h1>Enter File Path for Analysis</h1>
            <form action="/analyze" method="post">
                <label for="file_path">File Path:</label>
                <input type="text" id="file_path" name="file_path" required>
                <button type="submit">Analyze</button>
            </form>
        </body>
        </html>
    """)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file_path: str = Form(...)) -> HTMLResponse:
    """Analyze a document from the provided file path."""
    analyzer = DocumentAnalyzer()

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

    try:
        analysis_result = analyzer.analyze_document(file_path)
        result_html = f"""
            <h1>Analysis Results for {analysis_result['filename']}</h1>
            <p><b>Overall Risk Score:</b> {analysis_result['overall_score']}%</p>
            <ul>
        """
        for i, clause in enumerate(analysis_result["clauses"], 1):
            result_html += f"<li><b>Clause {i}:</b> {clause['clause_text']} <br> Risk: {clause['risk_level']}</li>"
        result_html += "</ul>"
        return HTMLResponse(content=result_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
