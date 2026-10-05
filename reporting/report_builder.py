"""
Report Builder Module

Extracts report generation logic with code review checkpoints.
Enables safe, user-approved code execution within reports.

Provides a two-step workflow:
1. build() - Generate report structure and extract code blocks (no execution)
2. execute_code_block() - User approves and executes individual code blocks
3. finalize() - Assemble final HTML after all code is resolved

Example:
    >>> from report_builder import ReportBuilder
    >>> builder = ReportBuilder(llm, mcp, retriever)
    >>> report = builder.build("Portfolio analysis", "Analyze Q2 performance", chunks)
    >>> # Report has pending code blocks
    >>> for block in report.pending_code_blocks:
    ...     # User reviews and approves
    ...     block = builder.execute_code_block(block)
    >>> final_report = builder.finalize(report)
    >>> print(final_report.html)
"""

from __future__ import annotations

import ast
import base64
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4

from context_retriever import ContextRetriever, RetrievedChunk
from mcp_manager import MCPError, MCPManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CodeBlock:
    """Represents an executable code block within a report section."""
    
    block_id: str
    section_name: str  # which report section this code belongs to
    code: str  # extracted Python code
    status: str = "pending"  # pending | approved | skipped | executed | error
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    images: List[str] = field(default_factory=list)  # base64 encoded PNG strings
    include_in_report: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "block_id": self.block_id,
            "section_name": self.section_name,
            "code": self.code,
            "status": self.status,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "images": self.images,
            "include_in_report": self.include_in_report,
        }


@dataclass
class ReportSection:
    """Represents a single section of a report."""
    
    title: str
    content: str  # LLM-generated markdown text
    code_block: Optional[CodeBlock] = None  # if this section has executable code
    chart_base64: Optional[str] = None  # if a chart was generated
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "content": self.content,
            "code_block": self.code_block.to_dict() if self.code_block else None,
            "chart_base64": self.chart_base64,
        }


@dataclass
class ReportResult:
    """Complete report with sections and code blocks."""
    
    goal: str
    sections: List[ReportSection]
    pending_code_blocks: List[CodeBlock]  # code waiting for user approval
    html: Optional[str] = None
    ready_to_finalize: bool = False  # True when all code blocks are resolved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "goal": self.goal,
            "sections": [s.to_dict() for s in self.sections],
            "pending_code_blocks": [b.to_dict() for b in self.pending_code_blocks],
            "html": self.html,
            "ready_to_finalize": self.ready_to_finalize,
        }


# ============================================================================
# REPORT BUILDER
# ============================================================================

class ReportBuilder:
    """
    Generates professional reports with optional code execution and visualization.
    
    Implements a two-step workflow:
    1. build() - Generate report structure and extract code blocks (no execution)
    2. execute_code_block() - Execute individual code blocks after user approval
    3. finalize() - Assemble final HTML after all code is resolved
    
    Attributes:
        llm: Language model for report generation
        mcp: Tool manager for code execution
        retriever: Context retriever for document search
    """
    
    # Blocked modules for security (file I/O, network, subprocess, etc.)
    BLOCKED_MODULES = {
        "os", "subprocess", "sys", "socket", "requests", "urllib", "httpx",
        "pathlib", "shutil", "glob", "importlib", "ctypes", "pickle",
        "__import__", "eval", "exec", "compile"
    }
    
    # Allowed modules for data analysis and visualization
    ALLOWED_MODULES = {
        "pandas", "numpy", "matplotlib", "plotly", "math", "statistics",
        "datetime", "json", "re", "collections", "itertools", "functools",
        "csv", "io", "textwrap", "operator", "bisect", "heapq", "decimal"
    }
    
    def __init__(
        self,
        llm: Any,
        mcp: MCPManager,
        retriever: ContextRetriever,
    ):
        """
        Initialize ReportBuilder.
        
        Args:
            llm: Language model for report generation
            mcp: Tool manager (for python_execute)
            retriever: Context retriever for formatting document chunks
        """
        self.llm = llm
        self.mcp = mcp
        self.retriever = retriever
    
    def build(
        self,
        goal: str,
        query: str,
        context_chunks: List[RetrievedChunk],
    ) -> ReportResult:
        """
        Generate report structure and extract code blocks (without execution).
        
        Steps:
        1. Call LLM to generate report with JSON structure
        2. Parse JSON and extract sections
        3. Identify code blocks
        4. Return with pending code blocks (not executed)
        
        Args:
            goal: Report goal/title
            query: Query to address in report
            context_chunks: Retrieved document chunks for context
        
        Returns:
            ReportResult with sections and pending code blocks
        """
        logger.info(f"Building report for goal: {goal}")
        
        # Format context and build prompt
        context_str = self.retriever.format_for_prompt(context_chunks)
        
        build_prompt = (
            f"Generate a professional MFI (Microfinance Institution) report for this goal: {goal}\n\n"
            f"Query to address: {query}\n\n"
            f"Context from documents:\n{context_str}\n\n"
            "Structure your response as JSON with this schema:\n"
            "{\n"
            '  "sections": [\n'
            '    {"title": "Executive Summary", "content": "...", "has_code": false},\n'
            '    {"title": "Analysis", "content": "...", "has_code": true, "code": "import pandas..."},\n'
            '    {"title": "Recommendations", "content": "...", "has_code": false}\n'
            "  ]\n"
            "}\n\n"
            "For sections with has_code=true: write real, runnable Python code.\n"
            "Allowed imports: pandas, numpy, matplotlib, plotly, math, statistics, datetime, json, re, collections, itertools, functools.\n"
            "FORBIDDEN: os, subprocess, open(), requests, urllib, pathlib, pickle, eval, exec.\n"
            "Do NOT use file I/O or network calls.\n"
            "Reply ONLY with the JSON object, no markdown markers.\n"
        )
        
        logger.debug("Calling LLM to generate report structure")
        try:
            raw_response = self.llm.generate(build_prompt, max_tokens=3000)
            
            # Clean JSON response
            cleaned = raw_response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            cleaned = cleaned.strip()
            
            logger.debug(f"Parsing JSON response: {len(cleaned)} chars")
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse report JSON: {e}")
            # Fallback: return single section
            parsed = {
                "sections": [
                    {
                        "title": goal,
                        "content": "Report generation encountered an error. Please try again.",
                        "has_code": False,
                    }
                ]
            }
        
        # Build sections and extract code blocks
        sections = []
        pending_blocks = []
        
        for s in parsed.get("sections", []):
            section = ReportSection(
                title=s.get("title", "Untitled"),
                content=s.get("content", ""),
            )
            
            # Extract code block if present
            if s.get("has_code") and s.get("code"):
                code_block = CodeBlock(
                    block_id=str(uuid4()),
                    section_name=s.get("title", ""),
                    code=s.get("code", ""),
                    status="pending",
                )
                section.code_block = code_block
                pending_blocks.append(code_block)
                logger.debug(f"Extracted code block for section: {section.title}")
            
            sections.append(section)
        
        logger.info(f"Report built: {len(sections)} sections, {len(pending_blocks)} code blocks")
        
        return ReportResult(
            goal=goal,
            sections=sections,
            pending_code_blocks=pending_blocks,
            ready_to_finalize=(len(pending_blocks) == 0),
        )
    
    def execute_code_block(
        self,
        block: CodeBlock,
        edited_code: Optional[str] = None,
    ) -> CodeBlock:
        """
        Execute a single code block after user approval and review.
        
        Steps:
        1. Use edited code if provided, otherwise use original
        2. Parse code with AST for static analysis
        3. Check for blocked/disallowed imports
        4. Execute via python_execute tool
        5. Capture stdout/stderr/images
        
        Args:
            block: CodeBlock to execute
            edited_code: Optional edited code from user review
        
        Returns:
            Updated CodeBlock with execution results
        """
        logger.info(f"Executing code block: {block.block_id}")
        
        final_code = edited_code if edited_code else block.code
        
        # Static analysis: parse and check for dangerous imports
        logger.debug("Performing static analysis")
        try:
            tree = ast.parse(final_code)
        except SyntaxError as e:
            block.status = "error"
            block.stderr = f"Syntax Error: {str(e)}"
            logger.warning(f"Syntax error in code block: {e}")
            return block
        
        # Check imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split(".")[0]
                    if module_name in self.BLOCKED_MODULES:
                        block.status = "error"
                        block.stderr = f"Blocked: '{alias.name}' is not allowed"
                        logger.warning(f"Blocked import: {alias.name}")
                        return block
                    if module_name not in self.ALLOWED_MODULES:
                        block.status = "error"
                        block.stderr = f"Not allowed: '{alias.name}' is not in the approved list"
                        logger.warning(f"Disallowed import: {alias.name}")
                        return block
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split(".")[0]
                    if module_name in self.BLOCKED_MODULES:
                        block.status = "error"
                        block.stderr = f"Blocked: 'from {node.module}' is not allowed"
                        logger.warning(f"Blocked import: from {node.module}")
                        return block
                    if module_name not in self.ALLOWED_MODULES:
                        block.status = "error"
                        block.stderr = f"Not allowed: 'from {node.module}' is not in the approved list"
                        logger.warning(f"Disallowed import: from {node.module}")
                        return block
        
        # Execute via python_execute tool
        logger.debug("Calling python_execute tool")
        block.status = "running"
        
        try:
            result = self.mcp.call_tool(
                "python_execute",
                code=final_code,
                timeout=30,
            )
            
            block.stdout = result.get("stdout", "")
            block.stderr = result.get("stderr", "")
            block.status = "executed" if result.get("success") else "error"
            
            # Handle generated images
            if result.get("image_path"):
                try:
                    block.images = [self._encode_image_to_base64(result["image_path"])]
                    logger.debug(f"Encoded image: {result['image_path']}")
                except Exception as e:
                    logger.warning(f"Failed to encode image: {e}")
            
            logger.info(f"Code block executed: status={block.status}")
            
        except MCPError as e:
            block.status = "error"
            block.stderr = str(e)
            logger.error(f"MCP error executing code: {e}")
        except Exception as e:
            block.status = "error"
            block.stderr = str(e)
            logger.error(f"Unexpected error executing code: {e}")
        
        return block
    
    def finalize(self, report: ReportResult) -> ReportResult:
        """
        Assemble final HTML report after all code blocks are resolved.
        
        Called after all code blocks are approved/skipped/executed.
        Generates HTML from markdown sections and includes code outputs.
        
        Args:
            report: ReportResult with sections and resolved code blocks
        
        Returns:
            Updated ReportResult with HTML content
        """
        logger.info(f"Finalizing report: {report.goal}")
        
        html_parts = []
        
        # Header
        html_parts.append(f"<html><head><title>{self._escape_html(report.goal)}</title></head><body>")
        html_parts.append(f"<h1>{self._escape_html(report.goal)}</h1>")
        
        # Sections
        for section in report.sections:
            html_parts.append(f"<h2>{self._escape_html(section.title)}</h2>")
            
            # Convert markdown to HTML (basic conversion)
            html_content = self._markdown_to_html(section.content)
            html_parts.append(html_content)
            
            # Include code block output if executed
            if section.code_block:
                block = section.code_block
                
                if block.status == "executed" or block.status == "error":
                    html_parts.append("<div class='code-output'>")
                    
                    if block.stdout:
                        html_parts.append("<h4>Output:</h4>")
                        html_parts.append(f"<pre>{self._escape_html(block.stdout)}</pre>")
                    
                    if block.stderr:
                        html_parts.append("<h4 style='color:red'>Error:</h4>")
                        html_parts.append(f"<pre style='color:red'>{self._escape_html(block.stderr)}</pre>")
                    
                    for img_b64 in (block.images or []):
                        html_parts.append(f"<img src='data:image/png;base64,{img_b64}' style='max-width:100%;'>")
                    
                    html_parts.append("</div>")
        
        html_parts.append("</body></html>")
        
        report.html = "\n".join(html_parts)
        report.ready_to_finalize = True
        
        logger.info(f"Report finalized: {len(report.html)} bytes of HTML")
        return report
    
    def _encode_image_to_base64(self, image_path: str) -> str:
        """
        Encode image file to base64 string.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Base64 encoded string
        """
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""
    
    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
    
    def _markdown_to_html(self, markdown: str) -> str:
        """
        Convert basic markdown to HTML.
        
        Handles:
        - Headers: # -> <h3>, ## -> <h4>, etc.
        - Bold: **text** -> <strong>text</strong>
        - Italic: *text* -> <em>text</em>
        - Code: `code` -> <code>code</code>
        - Lists: - item -> <ul><li>item</li></ul>
        - Paragraphs
        """
        html = self._escape_html(markdown)
        
        # Headers
        html = re.sub(r"^### (.+)$", r"<h4>\1</h4>", html, flags=re.MULTILINE)
        html = re.sub(r"^## (.+)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
        html = re.sub(r"^# (.+)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        
        # Bold
        html = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", html)
        
        # Italic
        html = re.sub(r"\*(.+?)\*", r"<em>\1</em>", html)
        
        # Code (inline)
        html = re.sub(r"`(.+?)`", r"<code>\1</code>", html)
        
        # Paragraphs (double newline = paragraph break)
        paragraphs = html.split("\n\n")
        html = "".join(f"<p>{p.strip()}</p>" if p.strip() else "" for p in paragraphs)
        
        # Line breaks
        html = html.replace("\n", "<br>")
        
        return html
