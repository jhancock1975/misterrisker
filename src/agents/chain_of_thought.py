"""Chain of Thought (CoT) support for trading agents.

This module provides structured reasoning prompts and response parsing
to enable Chain of Thought reasoning in all trading agents.
"""

import re
from enum import Enum
from typing import Any


class ReasoningType(Enum):
    """Types of reasoning for different query contexts."""
    
    ANALYSIS = "analysis"
    DECISION = "decision"
    COMPARISON = "comparison"
    RISK_ASSESSMENT = "risk_assessment"
    GENERAL = "general"


class ChainOfThought:
    """Chain of Thought reasoning support for agents.
    
    Provides structured prompts for step-by-step reasoning and
    parsing of LLM responses to extract reasoning steps.
    
    Attributes:
        default_reasoning_type: Default type of reasoning to use
    """
    
    def __init__(self, default_reasoning_type: ReasoningType = ReasoningType.GENERAL):
        """Initialize ChainOfThought.
        
        Args:
            default_reasoning_type: Default reasoning type to use
        """
        self.default_reasoning_type = default_reasoning_type
        
        # Prompt templates for each reasoning type
        self._prompts = {
            ReasoningType.ANALYSIS: self._analysis_prompt_template(),
            ReasoningType.DECISION: self._decision_prompt_template(),
            ReasoningType.COMPARISON: self._comparison_prompt_template(),
            ReasoningType.RISK_ASSESSMENT: self._risk_assessment_prompt_template(),
            ReasoningType.GENERAL: self._general_prompt_template(),
        }
    
    def _general_prompt_template(self) -> str:
        """Get general reasoning prompt template."""
        return """You are a financial analysis assistant. Think through this step by step.

## User Question
{query}

## Available Data
{data}

{portfolio_section}

## Instructions
Please reason through this step by step:

1. **Understanding**: First, understand what the user is asking.
2. **Data Review**: Review the available data and identify key points.
3. **Analysis**: Analyze the data to form insights.
4. **Synthesis**: Combine your analysis into a coherent response.

## Your Response

### Reasoning Steps
[Provide your step-by-step reasoning here]

### Conclusion
[Provide your final answer or recommendation]

Note: This is for informational purposes only and should not be considered financial advice."""
    
    def _analysis_prompt_template(self) -> str:
        """Get analysis reasoning prompt template."""
        return """You are a financial analysis assistant. Analyze the following data step by step.

## User Question
{query}

## Available Data
{data}

{portfolio_section}

## Instructions
Perform a thorough analysis step by step:

1. **Context**: Understand the analysis context and goals.
2. **Key Metrics**: Identify and evaluate key financial metrics.
3. **Trends**: Look for patterns and trends in the data.
4. **Factors**: Consider both positive and negative factors.
5. **Summary**: Synthesize your analysis into clear insights.

## Your Response

### Reasoning Steps
[Provide your step-by-step analysis here]

### Conclusion
[Provide your analytical conclusions]

Note: This analysis is for informational purposes only."""
    
    def _decision_prompt_template(self) -> str:
        """Get decision reasoning prompt template."""
        return """You are a financial analysis assistant. Help evaluate this decision step by step.

## User Question
{query}

## Available Data
{data}

{portfolio_section}

## Instructions
Think through this decision carefully, step by step:

1. **Objective**: Clarify what decision needs to be made.
2. **Pros**: List the potential benefits and positive factors.
3. **Cons**: List the potential risks and negative factors.
4. **Alternatives**: Consider alternative options.
5. **Recommendation**: Weigh the factors and provide a reasoned recommendation.

## Your Response

### Reasoning Steps
[Provide your step-by-step decision analysis here]

### Conclusion
[Provide your recommendation with reasoning]

**Important Risk Warning**: This is not financial advice. Always consult with a qualified financial advisor before making investment decisions. Consider your personal financial situation and risk tolerance."""
    
    def _comparison_prompt_template(self) -> str:
        """Get comparison reasoning prompt template."""
        return """You are a financial analysis assistant. Compare the options step by step.

## User Question
{query}

## Available Data
{data}

{portfolio_section}

## Instructions
Perform a structured comparison step by step:

1. **Identify Items**: Clearly identify what is being compared.
2. **Criteria**: Establish comparison criteria (price, value, growth, risk, etc.).
3. **Individual Analysis**: Analyze each item against the criteria.
4. **Side-by-Side**: Compare the items directly.
5. **Verdict**: Provide a reasoned comparison verdict.

## Your Response

### Reasoning Steps
[Provide your step-by-step comparison here]

### Conclusion
[Provide your comparison verdict]

Note: This comparison is for informational purposes only."""
    
    def _risk_assessment_prompt_template(self) -> str:
        """Get risk assessment prompt template."""
        return """You are a financial risk analyst. Assess the risks step by step.

## User Question
{query}

## Available Data
{data}

{portfolio_section}

## Instructions
Conduct a thorough risk assessment step by step:

1. **Risk Identification**: Identify all potential risks.
2. **Risk Categories**: Categorize risks (market, company-specific, sector, macro).
3. **Risk Magnitude**: Assess the potential impact of each risk.
4. **Risk Probability**: Estimate the likelihood of each risk.
5. **Mitigation**: Suggest possible risk mitigation strategies.

## Your Response

### Reasoning Steps
[Provide your step-by-step risk assessment here]

### Conclusion
[Provide your overall risk assessment]

**Important**: All investments carry risk. This assessment is for informational purposes only and should not be considered financial advice."""
    
    def get_reasoning_prompt(
        self,
        query: str,
        data: dict[str, Any],
        reasoning_type: ReasoningType | None = None,
        portfolio_context: dict[str, Any] | None = None
    ) -> str:
        """Generate a reasoning prompt for the given query and data.
        
        Args:
            query: User's question or request
            data: Available data for analysis
            reasoning_type: Type of reasoning to apply
            portfolio_context: Optional portfolio context
        
        Returns:
            Formatted prompt string
        """
        reasoning_type = reasoning_type or self.default_reasoning_type
        template = self._prompts.get(reasoning_type, self._prompts[ReasoningType.GENERAL])
        
        # Format data section
        data_str = self._format_data(data)
        
        # Format portfolio section if provided
        portfolio_section = ""
        if portfolio_context:
            portfolio_section = self._format_portfolio_context(portfolio_context)
        
        return template.format(
            query=query,
            data=data_str,
            portfolio_section=portfolio_section
        )
    
    def _format_data(self, data: dict[str, Any]) -> str:
        """Format data dictionary for prompt inclusion.
        
        Args:
            data: Data dictionary
        
        Returns:
            Formatted string
        """
        if not data:
            return "No specific data available."
        
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"**{key}**:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"**{key}**: {', '.join(str(v) for v in value[:5])}")
                if len(value) > 5:
                    lines.append(f"  (and {len(value) - 5} more...)")
            else:
                lines.append(f"**{key}**: {value}")
        
        return "\n".join(lines)
    
    def _format_portfolio_context(self, portfolio: dict[str, Any]) -> str:
        """Format portfolio context for prompt inclusion.
        
        Args:
            portfolio: Portfolio context dictionary
        
        Returns:
            Formatted string
        """
        lines = ["## Current Portfolio Context"]
        for symbol, position in portfolio.items():
            if isinstance(position, dict):
                shares = position.get("shares", "N/A")
                avg_cost = position.get("avg_cost", "N/A")
                lines.append(f"- **{symbol}**: {shares} shares at avg cost ${avg_cost}")
            else:
                lines.append(f"- **{symbol}**: {position}")
        
        return "\n".join(lines)
    
    def parse_response(self, response: str) -> dict[str, Any]:
        """Parse an LLM response to extract reasoning steps and conclusion.
        
        Args:
            response: Raw LLM response string
        
        Returns:
            Dictionary with reasoning_steps, conclusion, and raw_response
        """
        result = {
            "raw_response": response,
            "reasoning_steps": [],
            "conclusion": ""
        }
        
        # Try to extract reasoning steps
        steps = self._extract_reasoning_steps(response)
        if steps:
            result["reasoning_steps"] = steps
        
        # Try to extract conclusion
        conclusion = self._extract_conclusion(response)
        if conclusion:
            result["conclusion"] = conclusion
        
        return result
    
    def _extract_reasoning_steps(self, response: str) -> list[str]:
        """Extract reasoning steps from response.
        
        Args:
            response: LLM response string
        
        Returns:
            List of reasoning steps
        """
        steps = []
        
        # Look for numbered steps (1. 2. 3. etc)
        numbered_pattern = r'^\s*(\d+)\.\s*(.+?)(?=^\s*\d+\.|$|\n\n)'
        matches = re.findall(numbered_pattern, response, re.MULTILINE | re.DOTALL)
        if matches:
            steps = [match[1].strip() for match in matches]
        
        # Also look for steps in "Reasoning Steps" section
        reasoning_section = re.search(
            r'(?:Reasoning Steps|## Reasoning|### Reasoning Steps?)\s*(.*?)(?=##|###|\n\n\n|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if reasoning_section:
            section_text = reasoning_section.group(1)
            # Extract steps from this section
            section_steps = re.findall(r'^\s*[-*]?\s*(?:\d+[.)]?)?\s*(.+)$', section_text, re.MULTILINE)
            if section_steps:
                steps = [s.strip() for s in section_steps if s.strip() and len(s.strip()) > 10]
        
        return steps
    
    def _extract_conclusion(self, response: str) -> str:
        """Extract conclusion from response.
        
        Args:
            response: LLM response string
        
        Returns:
            Conclusion text
        """
        # Look for explicit conclusion section
        conclusion_match = re.search(
            r'(?:Conclusion|## Conclusion|### Conclusion)\s*(.*?)(?=##|###|\n\n\n|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if conclusion_match:
            return conclusion_match.group(1).strip()
        
        # Look for "In conclusion" or "To summarize"
        summary_match = re.search(
            r'(?:In conclusion|To summarize|Overall|In summary)[,:]?\s*(.*?)(?=\n\n|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if summary_match:
            return summary_match.group(1).strip()
        
        return ""
    
    def detect_reasoning_type(self, query: str, llm: Any = None) -> ReasoningType:
        """Detect the appropriate reasoning type from a query using LLM.
        
        Args:
            query: User query string
            llm: Language model instance (required)
        
        Returns:
            Detected ReasoningType
        """
        if not llm:
            # If no LLM provided, default to GENERAL
            return ReasoningType.GENERAL
        
        import json
        
        system_prompt = """Analyze the user's query and determine the type of reasoning needed.

Types:
- DECISION: User is asking whether to buy/sell, asking for recommendations, or needs a decision
- COMPARISON: User is comparing two or more options, asking which is better
- RISK_ASSESSMENT: User is asking about risk, volatility, safety, or downside
- ANALYSIS: User wants detailed analysis, evaluation, or assessment
- GENERAL: General question that doesn't fit the above

Return JSON only: {"reasoning_type": "DECISION|COMPARISON|RISK_ASSESSMENT|ANALYSIS|GENERAL"}"""

        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ])
            
            content = response.content
            if isinstance(content, list):
                content = content[0].get("text", "") if content else ""
            
            result = json.loads(content)
            reasoning_type_str = result.get("reasoning_type", "GENERAL").upper()
            
            type_map = {
                "DECISION": ReasoningType.DECISION,
                "COMPARISON": ReasoningType.COMPARISON,
                "RISK_ASSESSMENT": ReasoningType.RISK_ASSESSMENT,
                "ANALYSIS": ReasoningType.ANALYSIS,
                "GENERAL": ReasoningType.GENERAL,
            }
            
            return type_map.get(reasoning_type_str, ReasoningType.GENERAL)
            
        except Exception:
            return ReasoningType.GENERAL
