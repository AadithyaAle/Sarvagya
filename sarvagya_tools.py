import json
import os
from duckduckgo_search import DDGS

# --- Tool 1: The Knowledge Retrieval (Web Search) ---
def search_web(query: str):
    """Searches the web for technical information, datasheets, or debugging help."""
    print(f"\n[🔍 Searching Web for: '{query}']")
    try:
        results = DDGS().text(query, max_results=3)
        if not results: return "No results found."
        return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
    except Exception as e:
        return f"Search failed: {e}"

# --- Tool 2: The Project Tracker (JSON Todo List) ---
TODO_FILE = "sarvagya_tasks.json"

def update_todo(task: str, status: str):
    """Updates the project task list. Status can be 'pending', 'in_progress', or 'done'."""
    print(f"\n[📝 Updating Task: '{task}' -> {status}]")
    tasks = {}
    if os.path.exists(TODO_FILE):
        with open(TODO_FILE, 'r') as f: tasks = json.load(f)
    tasks[task] = status
    with open(TODO_FILE, 'w') as f: json.dump(tasks, f, indent=4)
    return f"Task '{task}' marked as {status}."

# --- Tool 3: The Auto-Coder (File Creator) ---
def create_file(filename: str, content: str):
    """
    Creates a new code file on the user's computer with the specified content.
    Args:
        filename: The name of the file (e.g., 'main.cpp', 'script.py').
        content: The actual code or text to write inside the file.
    """
    print(f"\n[💾 Writing new file: '{filename}']")
    try:
        with open(filename, 'w') as f:
            f.write(content)
        return f"Successfully created {filename} on the user's machine."
    except Exception as e:
        return f"Failed to create file: {e}"

# --- Tool Registry ---
tool_registry = {
    "search_web": search_web,
    "update_todo": update_todo,
    "create_file": create_file  # <-- Added the new tool here
}