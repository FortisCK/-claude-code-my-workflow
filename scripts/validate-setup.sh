#!/usr/bin/env bash
# =============================================================================
# validate-setup.sh - Verify local tools for the CATHACTION participant repo
# =============================================================================

set -uo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

pass=0
warn=0
fail=0

echo ""
echo -e "${BOLD}Validating CATHACTION participant workflow setup...${RESET}"
echo ""

check_required() {
    local name="$1"
    local cmd="$2"
    local install_url="$3"
    if command -v "$cmd" >/dev/null 2>&1; then
        echo -e "  ${GREEN}OK${RESET} $name found: $("$cmd" --version 2>&1 | head -n1)"
        pass=$((pass + 1))
    else
        echo -e "  ${RED}MISSING${RESET} $name - install: ${install_url}"
        fail=$((fail + 1))
    fi
}

check_recommended() {
    local name="$1"
    local cmd="$2"
    local install_url="$3"
    local why="$4"
    if command -v "$cmd" >/dev/null 2>&1; then
        echo -e "  ${GREEN}OK${RESET} $name found: $("$cmd" --version 2>&1 | head -n1)"
        pass=$((pass + 1))
    else
        echo -e "  ${YELLOW}WARN${RESET} $name not found - ${why}. Install: ${install_url}"
        warn=$((warn + 1))
    fi
}

echo -e "${BOLD}Required tools:${RESET}"
check_required "git" "git" "https://git-scm.com/downloads"
check_required "Python 3" "python3" "https://python.org"
echo ""

echo -e "${BOLD}Recommended for CATHACTION:${RESET}"
check_recommended "Docker" "docker" "https://docs.docker.com/get-docker/" "required once packaging the challenge submission"
check_recommended "Claude Code" "claude" "https://claude.ai/install" "useful for the original workflow commands"
check_recommended "XeLaTeX" "xelatex" "https://tug.org/texlive/ or https://tug.org/mactex/" "needed only for Beamer slides"
check_recommended "Quarto" "quarto" "https://quarto.org/docs/get-started/" "needed only for Quarto/web slides and guide rendering"
check_recommended "R" "R" "https://www.r-project.org/" "needed only for R scripts"
check_recommended "GitHub CLI" "gh" "https://cli.github.com/" "useful for PR workflow"
echo ""

echo -e "${BOLD}Git configuration:${RESET}"
if command -v git >/dev/null 2>&1; then
    git_name=$(git config user.name 2>/dev/null || true)
    git_email=$(git config user.email 2>/dev/null || true)
    if [ -n "$git_name" ] && [ -n "$git_email" ]; then
        echo -e "  ${GREEN}OK${RESET} git user: $git_name <$git_email>"
        pass=$((pass + 1))
    else
        echo -e "  ${YELLOW}WARN${RESET} git user.name / user.email not set"
        warn=$((warn + 1))
    fi
fi
echo ""

echo -e "${BOLD}Workflow scripts:${RESET}"
if python3 -m py_compile scripts/check-surface-sync.py scripts/check-palette-sync.py scripts/quality_score.py >/dev/null 2>&1; then
    echo -e "  ${GREEN}OK${RESET} Python workflow scripts compile"
    pass=$((pass + 1))
else
    echo -e "  ${RED}MISSING${RESET} One or more Python workflow scripts failed to compile"
    fail=$((fail + 1))
fi

palette_script="$(dirname "$0")/check-palette-sync.sh"
if [ -x "$palette_script" ]; then
    if "$palette_script" >/dev/null 2>&1; then
        echo -e "  ${GREEN}OK${RESET} Beamer/Quarto palette sync check passes"
        pass=$((pass + 1))
    else
        echo -e "  ${YELLOW}WARN${RESET} Palette drift - run ./scripts/check-palette-sync.sh"
        warn=$((warn + 1))
    fi
else
    echo -e "  ${YELLOW}WARN${RESET} scripts/check-palette-sync.sh missing or not executable"
    warn=$((warn + 1))
fi
echo ""

echo -e "${BOLD}Summary:${RESET} ${GREEN}${pass} passed${RESET}, ${YELLOW}${warn} warnings${RESET}, ${RED}${fail} failed${RESET}"
echo ""

if [ "$fail" -gt 0 ]; then
    echo -e "${RED}Setup has blocking issues.${RESET}"
    echo "Install the missing required tools above and re-run this script."
    exit 1
fi

echo -e "${GREEN}Core setup looks usable.${RESET}"
echo "Next project step: choose the ML stack, scaffold tests, and define the first CATHACTION baseline plan."
echo ""
exit 0
