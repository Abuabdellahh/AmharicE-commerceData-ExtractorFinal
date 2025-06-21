import os
import subprocess
from pathlib import Path

def compile_latex_report():
    """Compile the LaTeX report into a PDF"""
    # Create a temporary directory for compilation
    temp_dir = Path("temp_report")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy the report.tex file
    report_path = Path("report.tex")
    temp_report_path = temp_dir / "report.tex"
    
    # Copy the file to temp directory
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    with open(temp_report_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Compile the LaTeX report
    try:
        # First pass
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], 
                      cwd=str(temp_dir),
                      capture_output=True)
        
        # Second pass for references and table of contents
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "report.tex"], 
                      cwd=str(temp_dir),
                      capture_output=True)
        
        # Move the final PDF to the root directory
        pdf_path = temp_dir / "report.pdf"
        if pdf_path.exists():
            os.replace(pdf_path, Path("report.pdf"))
            print("Report compiled successfully!")
        else:
            print("Error: PDF file not generated")
            
    except Exception as e:
        print(f"Error compiling report: {e}")
    
    # Clean up temporary files
    for file in temp_dir.glob("report.*"):
        if file.suffix != ".tex":
            file.unlink()
    
    # Remove empty temp directory
    if not any(temp_dir.iterdir()):
        temp_dir.rmdir()

if __name__ == "__main__":
    compile_latex_report()
