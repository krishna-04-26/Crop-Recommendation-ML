#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
    interpreter=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    interpreter="python3"
else
    echo "Install Python 3.10 or later from python.org, then double-click this file again."
    read -r -p "Press Enter to close. "
    exit 1
fi
"$interpreter" launch.py
status=$?
if [ "$status" -ne 0 ]; then
    read -r -p "Press Enter to close. "
fi
exit "$status"
