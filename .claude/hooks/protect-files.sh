#!/bin/bash
# Block accidental edits to protected files
# Customize PROTECTED_PATTERNS below for your project
INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name')
FILE=""

# Extract file path based on tool type
if [ "$TOOL" = "Edit" ] || [ "$TOOL" = "Write" ]; then
  FILE=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
fi

# No file path = not a file operation, allow
if [ -z "$FILE" ]; then
  exit 0
fi

# ============================================================
# CUSTOMIZE: Add patterns for files you want to protect
# Uses basename matching — add full paths for more precision
# ============================================================
PROTECTED_PATTERNS=(
  "Bibliography_base.bib"
  "settings.json"
)

# Protect all thesis source files (read-only by default)
if [[ "$FILE" == *"/Thesis/"* ]]; then
  echo "Protected: Thesis source files are read-only. Ask the advisor for permission to edit, or remove protection in .claude/hooks/protect-files.sh" >&2
  exit 2
fi

BASENAME=$(basename "$FILE")
for PATTERN in "${PROTECTED_PATTERNS[@]}"; do
  if [[ "$BASENAME" == "$PATTERN" ]]; then
    echo "Protected file: $BASENAME. Edit manually or remove protection in .claude/hooks/protect-files.sh" >&2
    exit 2
  fi
done

exit 0
