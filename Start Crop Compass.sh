#!/bin/sh
cd "$(dirname "$0")" || exit 1
if [ -x ".venv/bin/python" ]; then
    interpreter=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    interpreter="python3"
else
    echo "Install Python 3.10+ and your distribution's python3-venv package."
    read -r answer
    exit 1
fi
"$interpreter" launch.py
status=$?
if [ "$status" -ne 0 ]; then
    echo "Press Enter to close."
    read -r answer
fi
exit "$status"
