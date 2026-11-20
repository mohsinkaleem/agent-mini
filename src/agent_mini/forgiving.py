"""Small-model fix: accept the tool and argument names models use instead of ours.

Asked to call ``edit_file(path, old_text, new_text)``, a 4B model will often send
``str_replace(file_path=…, old_string=…, new_string=…)``, or ``"10"`` where the
schema says the number 10. Rejecting that costs a round trip and often derails
the model, so the common variants are mapped to ours before a tool runs.

You can skip this file on a first read: a model that follows the schema never
needs it.
"""

from __future__ import annotations

import json

# Tool names models use for our tools (and our own names before 0.5). They also
# keep old tools.confirm lists working.
TOOL_ALIASES = {
    "shell_exec": "shell", "bash": "shell", "run_shell": "shell", "run_command": "shell",
    "execute_command": "shell", "terminal": "shell",
    "code_edit": "edit_file", "str_replace": "edit_file", "replace_in_file": "edit_file",
    "create_file": "write_file", "list_directory": "list_files", "find_files": "list_files",
    "ls": "list_files", "grep": "search_files", "memory_store": "remember", "memory_recall": "recall",
}

# Argument names models use instead of ours. Applied only when the tool has no
# parameter of that name.
ARG_ALIASES = {
    "file_path": "path", "filepath": "path", "filename": "path", "file": "path", "directory": "path",
    "dir": "path", "cmd": "command", "old_string": "old_text", "new_string": "new_text",
    "old_str": "old_text", "new_str": "new_text", "contents": "content", "text": "content",
    "glob": "pattern", "pattern": "query", "regex": "query", "search": "query",
}


def normalize_arguments(params: dict, arguments: dict) -> dict:
    """Map aliased argument names to ours and coerce values to the declared types.

    *params* is the tool's JSON schema ``parameters``.
    """
    props = params.get("properties") or {}
    out: dict = {}
    for key, value in arguments.items():
        # "new_text": null means "not given", never the text "None".
        if value is None:
            continue
        if key not in props and ARG_ALIASES.get(key) in props:
            key = ARG_ALIASES[key]
        # Both "path" and "file_path" were sent: the first one wins.
        if key in out:
            continue
        kind = props.get(key, {}).get("type")
        if kind == "integer" and isinstance(value, str) and value.strip().lstrip("-").isdigit():
            value = int(value)  # "10" becomes 10
        elif kind == "string" and not isinstance(value, str):
            # A JSON object sent as file content is written out as JSON.
            value = json.dumps(value, indent=2) if isinstance(value, (dict, list)) else str(value)
        out[key] = value
    return out
