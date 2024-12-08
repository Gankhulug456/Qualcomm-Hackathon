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
import re
from fastapi import File, UploadFile
import tempfile
from fastapi.responses import FileResponse



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

    def read_docx(self, file_path):
        """Read content from a DOCX file"""
        try:
            doc = Document(file_path)
            full_text = []
            clause_num = 0
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    # Explicitly mark the start of a new clause when text is all caps, which often signifies a heading
                    if text.isupper():
                        full_text.append(f"\n{clause_num}. " + text)
                        clause_num += 1
                    else:
                        full_text.append(text)
            return ''.join(full_text)
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
        """Break document into clauses based on common numbering patterns."""
        # Split by common clause numbering patterns
        clauses = []
        
        # Split by numbered patterns like "1.", "1.1", "(1)", "a.", "A."
        lines = document_text.split('\n')
        current_clause = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for common clause number patterns
            if (
                re.match(r'^\d+\.|\(\d+\)|[A-Za-z]\.|\d+\.\d+', line) or 
                re.match(r'^Section\s+\d+', line, re.IGNORECASE)
            ):
                if current_clause:
                    clauses.append(' '.join(current_clause))
                    current_clause = []
                current_clause.append(line)
            else:
                current_clause.append(line)
        
        # Add the last clause if exists
        if current_clause:
            clauses.append(' '.join(current_clause))

        if clauses:
            return clauses[1:]
        else:
            return [document_text]
            
    def analyze_clause(self, clause_text: str) -> Dict[str, Union[str, Any]]:
        """Analyze a single clause for risks."""
        prompt = f"""Analyze the following contract clause for risk level based on these examples:

        HIGH RISK examples:
        - Unlimited liability clauses
        - Complete waivers of rights
        - Automatic renewal with price increases
        - Unilateral contract changes

        MEDIUM RISK examples:
        - Late payment penalties
        - Maintenance responsibilities
        - Notice requirements
        - Standard termination clauses

        LOW RISK examples:
        - Basic contact information
        - Standard business hours
        - Regular payment schedules
        - Standard definitions

        Analyze this clause:
        {clause_text}

        Provide only:
        1. Risk level: [Low/Medium/High]
        2. Brief reason (max 30 word for high and medium high levels but for low level ones 0 words)"""

        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_ID", "model-identifier"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent responses
            max_tokens=100,   # Limit response length
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        analysis = response.choices[0].message.content
        return {
            "clause_text": clause_text,
            "analysis": analysis,
            "risk_level": self._extract_risk_level(analysis)
        }
    
    def generate_doc_summary(self, document_text):
        """Summarize the document as a whole."""
        prompt = f"""Summarize and Analyze the following contract for risk level based on these examples:

        HIGH RISK examples:
        - Unlimited liability clauses
        - Complete waivers of rights
        - Automatic renewal with price increases
        - Unilateral contract changes

        MEDIUM RISK examples:
        - Late payment penalties
        - Maintenance responsibilities
        - Notice requirements
        - Standard termination clauses

        LOW RISK examples:
        - Basic contact information
        - Standard business hours
        - Regular payment schedules
        - Standard definitions

        Analyze this document:
        {document_text}

        Provide a few-sentence summary of the document and draw conclusions, with examples, of which party the document benefits."""

        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_ID", "model-identifier"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more consistent responses
            max_tokens=300,   # Limit response length
            presence_penalty=0.1,
            frequency_penalty=0.1
        )
        
        summary = response.choices[0].message.content
        return summary

    def _extract_risk_level(self, analysis_text: str) -> str:
        """Extract risk level from analysis text."""
        lower_text = analysis_text.lower()
        if "high risk" in lower_text or "high" in lower_text:
            return "High"
        elif "medium risk" in lower_text or "medium" in lower_text:
            return "Medium"
        return "Low"

    def calculate_risk_score(self, analyzed_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate an optimized risk score focusing on critical factors."""
        if not analyzed_clauses:
            return {"overall_score": 0, "risk_breakdown": {}}

        # Simplified weights focusing on most critical clauses
        critical_terms = {
            # High Risk Terms (2.0)
            "indemnification": 2.0,
            "liability": 2.0,
            "warranty": 2.0,
            "damages": 2.0,
            "termination": 2.0,
            
            # Medium-High Risk Terms (1.5)
            "confidential": 1.5,
            "intellectual property": 1.5,
            "compliance": 1.5,
            "penalty": 1.5,
            "default": 1.5,
            
            # Medium Risk Terms (1.2)
            "payment": 1.2,
            "modification": 1.2,
            "notice": 1.2,
            "jurisdiction": 1.2,
            "renewal": 1.2
        }
        
        risk_weights = {"High": 3, "Medium": 2, "Low": 1}
        risk_breakdown = {"High": 0, "Medium": 0, "Low": 0}
        
        total_score = 0
        clause_count = len(analyzed_clauses)
        
        for clause in analyzed_clauses:
            # Get base risk score
            risk_level = clause["risk_level"]
            base_score = risk_weights.get(risk_level, 1)
            
            # Apply multiplier for critical terms
            multiplier = 1.0
            clause_text_lower = clause["clause_text"].lower()
            for term, weight in critical_terms.items():
                if term in clause_text_lower:
                    multiplier = max(multiplier, weight)
                    break
            
            total_score += base_score * multiplier
            risk_breakdown[risk_level] += 1
        
        # Calculate normalized score (0-100)
        max_possible_score = clause_count * 3 * 2.0  # maximum risk * maximum multiplier
        overall_score = (total_score / max_possible_score) * 100 if max_possible_score > 0 else 0
        
        return {
            "overall_score": round(overall_score, 2),
            "risk_breakdown": risk_breakdown
        }

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """Main method to analyze a document."""
        document_text = self.read_file(file_path)
        clauses = self.extract_clauses(document_text)
        analyzed_clauses = [self.analyze_clause(clause) for clause in clauses]
        risk_data = self.calculate_risk_score(analyzed_clauses)
        summary = self.generate_doc_summary(document_text)
        analysis_result = {
            "filename": os.path.basename(file_path),
            "clauses": analyzed_clauses,
            "overall_score": risk_data["overall_score"],
            "risk_breakdown": risk_data["risk_breakdown"],
        }
        return analysis_result, summary


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Render the HTML template."""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/styles/styles.css", response_class=FileResponse)
async def get_css():
    """Serve the CSS file."""
    return FileResponse(os.path.join("styles", "styles.css"))

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(file: UploadFile = File(...)) -> HTMLResponse:
    """Analyze an uploaded document."""
    # Define allowed MIME types
    allowed_types = [
        "application/pdf",
        "text/plain",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ]

    # Check file type
    content_type = file.content_type
    if content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {content_type}. Allowed types are: PDF, TXT, DOC, and DOCX"
        )

    analyzer = DocumentAnalyzer()

    try:
        # Create a temporary file to store the upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        analysis_result, summary = analyzer.analyze_document(temp_path)
        
        # Clean up the temporary file
        os.unlink(temp_path)

        result_html = f"""
            <h1>Analysis Results for {analysis_result['filename']}</h1>
            <p><b>Overall Risk Score:</b> {analysis_result['overall_score']}%</p>
            <ul>
        """
        for i, clause in enumerate(analysis_result["clauses"], 1):
            result_html += f"<li><b>Clause {i}:</b> {clause['clause_text']} <br> Risk: {clause['risk_level']}</li>"
        
        result_html += f'<li><b>Summary:</b> {summary}</li'
        result_html += "</ul>"
        return HTMLResponse(content=result_html)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
