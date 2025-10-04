import os
from dotenv import load_dotenv
from datetime import datetime
import sympy as sp
import numpy as np
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import tempfile
import subprocess

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, field_validator


# Security Utilities (keeping the same secure functions)
def safe_path(path):
    """Prevent directory traversal, allow only files within a specified directory."""
    base_dir = os.path.abspath("safe_files")
    os.makedirs(base_dir, exist_ok=True)  # Ensure directory exists
    abs_path = os.path.abspath(os.path.join(base_dir, path))
    if not abs_path.startswith(base_dir):
        raise ValueError("Unsafe file path detected!")
    return abs_path


def is_url_safe(url):
    """Only allow http/https, block suspicious domains"""
    if not isinstance(url, str):
        return False
    return url.startswith("http://") or url.startswith("https://")


# Pydantic Input Schemas for each tool
class SystemTimeInput(BaseModel):
    """No input required for system time"""
    pass


class MathToolInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression to evaluate using SymPy")
    
    @field_validator('expression')
    def validate_expression(cls, v):
        if len(v) > 200:
            raise ValueError("Expression too long")
        return v


class PythonExecutorInput(BaseModel):
    code: str = Field(..., description="Python code to execute in secure sandbox")
    
    @field_validator('code')
    def validate_code(cls, v):
        if len(v) > 5000:
            raise ValueError("Code too long")
        dangerous_imports = ['os', 'subprocess', 'sys', '__import__', 'eval', 'exec']
        if any(danger in v for danger in dangerous_imports):
            raise ValueError("Potentially dangerous code detected")
        return v


class WebSearchInput(BaseModel):
    query: str = Field(..., description="Search query string")
    
    @field_validator('query')
    def validate_query(cls, v):
        if len(v) > 128:
            raise ValueError("Query too long")
        return v


class WebScraperInput(BaseModel):
    url: str = Field(..., description="URL to scrape (must be http/https)")
    
    @field_validator('url')
    def validate_url(cls, v):
        if not is_url_safe(v):
            raise ValueError("Unsafe URL")
        return v


class FileHandlerInput(BaseModel):
    filepath: str = Field(..., description="File path within safe directory")
    mode: str = Field(default='r', description="File mode: 'r' for read, 'w' for write")
    content: Optional[str] = Field(None, description="Content to write (required for write mode)")
    
    @field_validator('filepath')
    def validate_filepath(cls, v):
        if '..' in v or v.startswith('/'):
            raise ValueError("Invalid file path")
        return v


class DataVisualizerInput(BaseModel):
    data: Dict[str, list] = Field(..., description="Data dictionary with 'x' and 'y' keys containing lists")
    plot_type: str = Field(default='line', description="Plot type: 'line' or 'bar'")
    title: str = Field(default='Chart', description="Chart title")
    xlabel: str = Field(default='', description="X-axis label")
    ylabel: str = Field(default='', description="Y-axis label")


class EmailInput(BaseModel):
    to_address: str = Field(..., description="Recipient email address")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body content")
    
    @field_validator('to_address')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError("Invalid email address")
        return v


class ShellExecutorInput(BaseModel):
    command: str = Field(..., description="Shell command to execute (limited whitelist)")


class TextToSpeechInput(BaseModel):
    text: str = Field(..., description="Text to convert to speech")
    filename: str = Field(default='output.mp3', description="Output filename")
    
    @field_validator('text')
    def validate_text(cls, v):
        if len(v) > 500:
            raise ValueError("Text too long")
        return v


class SpeechToTextInput(BaseModel):
    audio_file: str = Field(..., description="Path to audio file")


class TextFromImageInput(BaseModel):
    image_file: str = Field(..., description="Path to image file")


class MediaPlayerInput(BaseModel):
    media_file: str = Field(..., description="Path to media file")


class JavaScriptExecutorInput(BaseModel):
    js_code: str = Field(..., description="JavaScript code to execute")
    
    @field_validator('js_code')
    def validate_js(cls, v):
        dangerous_patterns = ['require(', 'process', 'fs.', '__dirname', '__filename']
        if any(pattern in v for pattern in dangerous_patterns):
            raise ValueError("Unsafe JavaScript detected")
        return v


class DataFetcherInput(BaseModel):
    url: str = Field(..., description="URL to download from")
    save_path: str = Field(..., description="Local path to save file")
    
    @field_validator('url')
    def validate_url(cls, v):
        if not is_url_safe(v):
            raise ValueError("Unsafe URL")
        return v


# BaseTool Implementations

class SystemTimeTool(BaseTool):
    name = "get_system_time"
    description = "Get current system date and time"
    args_schema = SystemTimeInput
    
    def _run(self, **kwargs) -> str:
        """Return system time (no user input allowed)."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MathTool(BaseTool):
    name = "math_tool"
    description = "Safely evaluate mathematical expressions using SymPy. Supports variables like x, basic arithmetic, and mathematical functions."
    args_schema = MathToolInput
    
    def _run(self, expression: str) -> Union[str, Dict[str, str]]:
        """Evaluates mathematical expressions safely, blocks code injection."""
        allowed_names = {"x": sp.symbols("x")}
        try:
            expr = sp.sympify(expression, locals=allowed_names)
            result = expr.evalf()
            return str(result)
        except Exception as e:
            return {"error": f"Math error: {str(e)}"}


class PythonExecutorTool(BaseTool):
    name = "python_executor"
    description = "Execute Python code in a secure sandbox with timeout. Dangerous imports are blocked."
    args_schema = PythonExecutorInput
    
    def _run(self, code: str) -> Dict[str, str]:
        """Execute Python code in a sandboxed subprocess."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmpfile:
            tmpfile.write(code)
            tmpfile_name = tmpfile.name
        
        try:
            result = subprocess.run(
                ["python", tmpfile_name],
                capture_output=True, text=True,
                timeout=5
            )
            if result.returncode != 0:
                return {"error": result.stderr.strip()}
            else:
                return {"output": result.stdout.strip()}
        except subprocess.TimeoutExpired:
            return {"error": "Code execution timeout"}
        except Exception as e:
            return {"error": str(e)}
        finally:
            os.remove(tmpfile_name)


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the web using external APIs. Returns search URLs."
    args_schema = WebSearchInput
    
    def _run(self, query: str) -> Union[list, Dict[str, str]]:
        """Search using external API."""
        if not isinstance(query, str) or len(query) > 128:
            return {"error": "Bad query"}
        # Placeholder - replace with actual search API
        return [f"https://duckduckgo.com/?q={requests.utils.quote(query)}"]


class WebScraperTool(BaseTool):
    name = "web_scraper"
    description = "Scrape text content from web pages. Only returns plain text, strips HTML tags."
    args_schema = WebScraperInput
    
    def _run(self, url: str) -> Union[str, Dict[str, str]]:
        """Scrape HTML from URL, return only text."""
        try:
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                return {"error": f"HTTP error: {response.status_code}"}
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text().strip()
        except Exception as e:
            return {"error": str(e)}


class FileHandlerTool(BaseTool):
    name = "file_handler"
    description = "Safely read from or write to files within a secure directory. Prevents directory traversal attacks."
    args_schema = FileHandlerInput
    
    def _run(self, filepath: str, mode: str = 'r', content: Optional[str] = None) -> Union[str, bool, Dict[str, str]]:
        """Handle file operations safely."""
        try:
            safe_fp = safe_path(filepath)
            if mode == 'r':
                with open(safe_fp, 'r', encoding='utf-8') as f:
                    return f.read()
            elif mode == 'w' and content is not None:
                if len(content) > 1e6:  # 1MB max
                    return {"error": "Content too large"}
                with open(safe_fp, 'w', encoding='utf-8') as f:
                    f.write(content)
                    return True
            else:
                return {"error": "Invalid mode or missing content"}
        except Exception as e:
            return {"error": str(e)}


class DataVisualizerTool(BaseTool):
    name = "data_visualizer"
    description = "Create line or bar charts from data. Data must be provided as dict with 'x' and 'y' keys."
    args_schema = DataVisualizerInput
    
    def _run(self, data: Dict[str, list], plot_type: str = 'line', 
             title: str = 'Chart', xlabel: str = '', ylabel: str = '') -> Union[str, Dict[str, str]]:
        """Create charts from data."""
        import matplotlib.pyplot as plt
        
        if not (isinstance(data, dict) and 'x' in data and 'y' in data):
            return {"error": "Invalid data format"}
        
        try:
            if len(data['x']) > 1000 or len(data['y']) > 1000:
                return {"error": "Data too large"}
            
            plt.figure()
            if plot_type == 'line':
                plt.plot(data['x'], data['y'])
            elif plot_type == 'bar':
                plt.bar(data['x'], data['y'])
            else:
                return {"error": "Invalid plot type"}
            
            plt.title(title)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.tight_layout()
            plt.show()
            return "Chart displayed."
        except Exception as e:
            return {"error": str(e)}


# ... (continue with remaining tools)

# Tool Registry - All tools in one place
ALL_TOOLS = [
    SystemTimeTool(),
    MathTool(),
    PythonExecutorTool(),
    WebSearchTool(),
    WebScraperTool(),
    FileHandlerTool(),
    DataVisualizerTool(),
    # Add remaining tools...
]

# Create tools dictionary for easy access
TOOLS_DICT = {tool.name: tool for tool in ALL_TOOLS}
