from openai import OpenAI
from docx import Document
from dotenv import load_dotenv
import os
import json

class DocumentAnalyzer:
    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            base_url=os.getenv("API_BASE_URL", "http://localhost:1234/v1"),
            api_key=os.getenv("API_KEY", "lm-studio")
        )

    def read_docx(self, file_path):
        """Read content from a DOCX file"""
        try:
            doc = Document(file_path)
            full_text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            return full_text
        except Exception as e:
            print(f"Error reading document: {e}")
            return None

    def extract_clauses(self, document_text):
        """Break document into clauses using AI"""
        prompt = f"""
        Break the following contract into separate clauses:
        {document_text}
        Return only the separated clauses.
        """
        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_ID", "model-identifier"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content.split('\n\n')

    def analyze_clause(self, clause_text):
        """Analyze a single clause for risks"""
        prompt = f"""
        Analyze the following contract clause for risk level and provide:
        1. Risk level (Low, Medium, High)
        2. Brief explanation of the risk
        
        Clause: {clause_text}
        """
        
        response = self.client.chat.completions.create(
            model=os.getenv("MODEL_ID", "model-identifier"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "clause_text": clause_text,
            "analysis": analysis,
            "risk_level": self._extract_risk_level(analysis)
        }

    def _extract_risk_level(self, analysis_text):
        """Extract risk level from analysis text"""
        lower_text = analysis_text.lower()
        if "high risk" in lower_text or "high" in lower_text:
            return "High"
        elif "medium risk" in lower_text or "medium" in lower_text:
            return "Medium"
        return "Low"

    def analyze_document(self, file_path):
        """Main method to analyze a document"""
        try:
            # Read document
            print("Reading document...")
            document_text = self.read_docx(file_path)
            if not document_text:
                return None

            # Extract clauses
            print("Extracting clauses...")
            clauses = self.extract_clauses(document_text)

            # Analyze each clause
            print("Analyzing clauses...")
            analyzed_clauses = []
            for i, clause in enumerate(clauses, 1):
                print(f"Analyzing clause {i} of {len(clauses)}...")
                analysis = self.analyze_clause(clause)
                analyzed_clauses.append(analysis)

            # Create final analysis
            analysis_result = {
                "filename": os.path.basename(file_path),
                "full_text": document_text,
                "clauses": analyzed_clauses,
                "high_risk_count": sum(1 for c in analyzed_clauses if c["risk_level"] == "High"),
                "medium_risk_count": sum(1 for c in analyzed_clauses if c["risk_level"] == "Medium"),
                "low_risk_count": sum(1 for c in analyzed_clauses if c["risk_level"] == "Low")
            }

            # Save results
            output_path = f"{file_path}_analysis.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, indent=2, ensure_ascii=False)

            print(f"\nAnalysis saved to: {output_path}")
            return analysis_result

        except Exception as e:
            print(f"Error analyzing document: {e}")
            return None

    def get_high_risk_clauses(self, analysis_result):
        """Extract only high risk clauses from analysis"""
        if not analysis_result or "clauses" not in analysis_result:
            return []
        return [
            clause for clause in analysis_result["clauses"]
            if clause["risk_level"] == "High"
        ]

def main():
    # Create .env file if it doesn't exist
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("""API_BASE_URL=http://localhost:1234/v1
API_KEY=lm-studio
MODEL_ID=model-identifier""")
        print("Created .env file with default values")

    # Initialize analyzer
    analyzer = DocumentAnalyzer()
    
    # Get document path from user
    file_path = input("Enter the path to your DOCX file: ")
    
    # Process document
    print("\nStarting document analysis...")
    result = analyzer.analyze_document(file_path)
    
    if result:
        # Get high risk clauses
        high_risk_clauses = analyzer.get_high_risk_clauses(result)
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total clauses: {len(result['clauses'])}")
        print(f"High risk clauses: {result['high_risk_count']}")
        print(f"Medium risk clauses: {result['medium_risk_count']}")
        print(f"Low risk clauses: {result['low_risk_count']}")
        
        # Print high risk details
        if high_risk_clauses:
            print("\nHigh Risk Clauses:")
            for i, clause in enumerate(high_risk_clauses, 1):
                print(f"\n{i}. Clause: {clause['clause_text']}")
                print(f"Analysis: {clause['analysis']}")
        else:
            print("\nNo high risk clauses found.")
    else:
        print("Analysis failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
