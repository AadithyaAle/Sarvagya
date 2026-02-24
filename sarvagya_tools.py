import json
import os

# --- Tool 1: The Project Tracker (JSON Todo List) ---
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

# --- Tool 2: The Auto-Coder (File Creator) ---
def create_file(filename: str, content: str):
    """
    Creates a new code file on the user's computer with the specified content.
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
    "update_todo": update_todo,
    "create_file": create_file 
}