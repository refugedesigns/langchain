from langchain.output_parsers import BaseOutputParser
import re
from typing import Any
import json
class NewAgentOutputParser(BaseOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Any:
        print("-" * 20)
        cleaned_output = text.strip()
        # Regex patterns to match action and action_input
        action_pattern = r'"action":\s*"([^"]*)"'
        action_input_pattern = r'"action_input":\s*"([^"]*)"'

        # Extracting first action and action_input values
        action = re.search(action_pattern, cleaned_output)
        action_input = re.search(action_input_pattern, cleaned_output)

        if action:
            action_value = action.group(1)
            print(f"First Action: {action_value}")
        else:
            print("Action not found")

        if action_input:
            action_input_value = action_input.group(1)
            print(f"First Action Input: {action_input_value}")
        else:
            print("Action Input not found")

        print("-" * 20)
        if action_value and action_input_value:
            return {"action": action_value, "action_input": action_input_value}

        # Problematic code left just in case
        if "```json" in cleaned_output:
            _, cleaned_output = cleaned_output.split("```json")
        if "```" in cleaned_output:
            cleaned_output, _ = cleaned_output.split("```")
        if cleaned_output.startswith("```json"):
            cleaned_output = cleaned_output[len("```json"):]
        if cleaned_output.startswith("```"):
            cleaned_output = cleaned_output[len("```"):]
        if cleaned_output.endswith("```"):
            cleaned_output = cleaned_output[: -len("```")]
        cleaned_output = cleaned_output.strip()
        response = json.loads(cleaned_output)
        return {"action": response["action"], "action_input": response["action_input"]}
