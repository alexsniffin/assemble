from typing import Dict, Any

from assemble.app.core.tools.adapter import ToolDetails


def build_schema(name: str, description: str, tool_parameters: Dict) -> Dict[str, Any]:
    tool_details = ToolDetails(
        tool_name=name,
        tool_description=description,
    )
    tool_schema = {**tool_details.model_dump(), **tool_parameters}
    tool_schema['properties']['tool_name'] = {'title': 'Name', 'type': 'string'}
    tool_schema['required'].append('tool_name')
    return tool_schema
