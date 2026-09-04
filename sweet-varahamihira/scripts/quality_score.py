#!/usr/bin/env python3
"""
Quality Scoring System for TRUST Paper Project

Calculates objective quality scores (0-100) based on defined rubrics.
Enforces quality gates: 80 (commit), 90 (PR), 95 (excellence).

Usage:
    python3 scripts/quality_score.py paper/main.tex
    python3 scripts/quality_score.py paper/main.tex --summary
    python3 scripts/quality_score.py paper/*.tex
    python3 scripts/quality_score.py scripts/python/plot_results.py
"""

import sys
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import re
import json

# ==============================================================================
# SCORING RUBRIC (from .claude/rules/quality-gates.md)
# ==============================================================================

PAPER_RUBRIC = {
    'critical': {
        'compilation_failure': {'points': 100, 'auto_fail': True},
        'undefined_citation': {'points': 15},
        'broken_figure_ref': {'points': 15},
        'equation_error': {'points': 10},
    },
    'major': {
        'overfull_hbox': {'points': 5},
        'notation_inconsistency': {'points': 3},
        'inconsistent_terminology': {'points': 3},
        'missing_figure_caption': {'points': 3},
    },
    'minor': {
        'table_formatting': {'points': 1},
        'long_lines': {'points': 1},
        'style_deviation': {'points': 1},
    }
}

PYTHON_SCRIPT_RUBRIC = {
    'critical': {
        'syntax_error': {'points': 100, 'auto_fail': True},
        'missing_imports': {'points': 10},
    },
    'major': {
        'no_seed': {'points': 5},
        'hardcoded_path': {'points': 5},
        'missing_figure_output': {'points': 5},
    },
    'minor': {
        'missing_type_hints': {'points': 1},
        'style_violation': {'points': 1},
    }
}

THRESHOLDS = {
    'commit': 80,
    'pr': 90,
    'excellence': 95
}

# ==============================================================================
# ISSUE DETECTION (Lightweight checks - full agents run separately)
# ==============================================================================

class IssueDetector:
    """Detect common issues for quality scoring."""

    @staticmethod
    def check_equation_overflow(content: str) -> List[int]:
        """Detect displayed equations with single lines likely to overflow.

        Flags equations only when a SINGLE LINE within a math block exceeds
        120 characters.
        """
        overflows = []
        lines = content.split('\n')
        in_math = False
        math_delim = None

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if '$$' in stripped and math_delim != 'env':
                if not in_math:
                    in_math = True
                    math_delim = '$$'
                    if stripped.count('$$') >= 2:
                        inner = stripped.split('$$')[1]
                        if len(inner.strip()) > 120:
                            overflows.append(i)
                        in_math = False
                        math_delim = None
                    continue
                else:
                    in_math = False
                    math_delim = None
                    continue

            env_begin = re.match(
                r'\\begin\{(equation|align|gather|multline|eqnarray)\*?\}', stripped
            )
            if env_begin and not in_math:
                in_math = True
                math_delim = 'env'
                continue

            if re.match(r'\\end\{(equation|align|gather|multline|eqnarray)\*?\}', stripped):
                in_math = False
                math_delim = None
                continue

            if in_math:
                code_part = line.split('%')[0] if '%' in line else line
                if len(code_part.strip()) > 120:
                    overflows.append(i)

        return overflows

    @staticmethod
    def check_broken_citations(content: str, bib_file: Path) -> List[str]:
        """Check for LaTeX citation keys not in bibliography."""
        cite_pattern = r'\\cite[a-z]*\{([^}]+)\}'
        cited_keys = set()
        for match in re.finditer(cite_pattern, content):
            keys = match.group(1).split(',')
            cited_keys.update(k.strip() for k in keys)

        if not bib_file.exists():
            return list(cited_keys)

        bib_content = bib_file.read_text(encoding='utf-8')
        bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib_content))

        broken = cited_keys - bib_keys
        return list(broken)

    @staticmethod
    def check_hardcoded_paths(content: str) -> List[int]:
        """Detect absolute paths in scripts."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            if re.search(r'["\'][/\\]|["\'][A-Za-z]:[/\\]', line):
                if not re.search(r'http:|https:|file://|/tmp/', line):
                    issues.append(i)

        return issues

    @staticmethod
    def check_latex_syntax(content: str) -> List[Dict]:
        """Check for common LaTeX syntax issues without compiling."""
        issues = []
        lines = content.split('\n')

        env_stack = []
        for i, line in enumerate(lines, 1):
            stripped = line.split('%')[0] if '%' in line else line

            for match in re.finditer(r'\\begin\{(\w+)\}', stripped):
                env_stack.append((match.group(1), i))

            for match in re.finditer(r'\\end\{(\w+)\}', stripped):
                env_name = match.group(1)
                if env_stack and env_stack[-1][0] == env_name:
                    env_stack.pop()
                elif env_stack:
                    issues.append({
                        'line': i,
                        'description': f'Mismatched environment: \\end{{{env_name}}} '
                                       f'but expected \\end{{{env_stack[-1][0]}}} '
                                       f'(opened at line {env_stack[-1][1]})',
                    })
                else:
                    issues.append({
                        'line': i,
                        'description': f'\\end{{{env_name}}} without matching \\begin',
                    })

        for env_name, line_num in env_stack:
            issues.append({
                'line': line_num,
                'description': f'Unclosed environment: \\begin{{{env_name}}} never closed',
            })

        return issues

    @staticmethod
    def check_overfull_hbox_risk(content: str) -> List[int]:
        """Detect lines in LaTeX source likely to cause overfull hbox."""
        issues = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            stripped = line.split('%')[0] if '%' in line else line

            if len(stripped.strip()) > 120:
                if stripped.strip().startswith('%'):
                    continue
                if re.match(r'\s*\\(includegraphics|input|bibliography|usepackage|bibliographystyle)', stripped):
                    continue
                issues.append(i)

        return issues

    @staticmethod
    def check_broken_figure_refs(content: str, paper_dir: Path) -> List[str]:
        """Check that all \\includegraphics references point to existing files."""
        broken = []
        pattern = r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}'
        for match in re.finditer(pattern, content):
            fig_path = match.group(1)
            # Try resolving relative to paper directory
            full_path = paper_dir / fig_path
            if not full_path.exists():
                # Also try from repo root
                repo_root = paper_dir.parent
                full_path = repo_root / fig_path
                if not full_path.exists():
                    broken.append(fig_path)
        return broken

    @staticmethod
    def check_python_syntax(filepath: Path) -> Tuple[bool, str]:
        """Check Python script for syntax errors using AST."""
        try:
            result = subprocess.run(
                ['python3', '-c', f'import ast; ast.parse(open("{filepath}").read())'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return False, result.stderr
            return True, ""
        except subprocess.TimeoutExpired:
            return False, "Syntax check timeout"
        except FileNotFoundError:
            return False, "python3 not installed"


# ==============================================================================
# QUALITY SCORER
# ==============================================================================

class QualityScorer:
    """Calculate quality scores for paper materials."""

    def __init__(self, filepath: Path, verbose: bool = False):
        self.filepath = filepath
        self.verbose = verbose
        self.score = 100
        self.issues = {
            'critical': [],
            'major': [],
            'minor': []
        }
        self.auto_fail = False

    def score_paper(self) -> Dict:
        """Score LaTeX paper quality."""
        content = self.filepath.read_text(encoding='utf-8')

        # Check for LaTeX syntax issues (without compiling)
        syntax_issues = IssueDetector.check_latex_syntax(content)
        if syntax_issues:
            for issue in syntax_issues:
                self.issues['critical'].append({
                    'type': 'compilation_failure',
                    'description': f'LaTeX syntax issue at line {issue["line"]}',
                    'details': issue['description'],
                    'points': 100
                })
            self.auto_fail = True
            self.score = 0
            return self._generate_report()

        # Check for undefined/broken citations
        bib_file = self.filepath.parent.parent / 'Bibliography_base.bib'
        if not bib_file.exists():
            bib_file = self.filepath.parent / 'Bibliography_base.bib'
        broken_citations = IssueDetector.check_broken_citations(content, bib_file)
        for key in broken_citations:
            self.issues['critical'].append({
                'type': 'undefined_citation',
                'description': f'Citation key not in bibliography: {key}',
                'details': 'Add to Bibliography_base.bib or fix key',
                'points': 15
            })
            self.score -= 15

        # Check for broken figure references
        broken_figs = IssueDetector.check_broken_figure_refs(content, self.filepath.parent)
        for fig_path in broken_figs:
            self.issues['critical'].append({
                'type': 'broken_figure_ref',
                'description': f'Figure not found: {fig_path}',
                'details': 'Check that the figure exists in figs/',
                'points': 15
            })
            self.score -= 15

        # Check for lines likely to cause overfull hbox
        overfull_lines = IssueDetector.check_overfull_hbox_risk(content)
        for line in overfull_lines:
            self.issues['major'].append({
                'type': 'overfull_hbox',
                'description': f'Potential overfull hbox at line {line}',
                'details': 'Line >120 chars may overflow column width',
                'points': 5
            })
            self.score -= 5

        # Check equation overflow
        equation_overflows = IssueDetector.check_equation_overflow(content)
        for line_num in equation_overflows:
            self.issues['critical'].append({
                'type': 'equation_error',
                'description': f'Potential equation overflow at line {line_num}',
                'details': 'Single equation line >120 chars likely to overflow',
                'points': 10
            })
            self.score -= 10

        self.score = max(0, self.score)
        return self._generate_report()

    def score_python(self) -> Dict:
        """Score Python script quality."""
        content = self.filepath.read_text(encoding='utf-8')

        # Check syntax
        is_valid, error = IssueDetector.check_python_syntax(self.filepath)
        if not is_valid:
            self.auto_fail = True
            self.issues['critical'].append({
                'type': 'syntax_error',
                'description': 'Python syntax error',
                'details': error[:200],
                'points': 100
            })
            self.score = 0
            return self._generate_report()

        # Check hardcoded paths
        path_issues = IssueDetector.check_hardcoded_paths(content)
        for line in path_issues:
            self.issues['major'].append({
                'type': 'hardcoded_path',
                'description': f'Hardcoded absolute path at line {line}',
                'details': 'Use relative paths',
                'points': 5
            })
            self.score -= 5

        # Check for random seed if randomness detected
        has_random = any(fn in content for fn in ['random.', 'np.random.', 'torch.manual_seed'])
        has_seed = any(s in content for s in ['random.seed', 'np.random.seed', 'torch.manual_seed'])
        if has_random and not has_seed:
            self.issues['major'].append({
                'type': 'no_seed',
                'description': 'Missing random seed for reproducibility',
                'details': 'Add random.seed() / np.random.seed() at script top',
                'points': 5
            })
            self.score -= 5

        self.score = max(0, self.score)
        return self._generate_report()

    def _generate_report(self) -> Dict:
        """Generate quality score report."""
        if self.auto_fail:
            status = 'FAIL'
            threshold = 'None (auto-fail)'
        elif self.score >= THRESHOLDS['excellence']:
            status = 'EXCELLENCE'
            threshold = 'excellence'
        elif self.score >= THRESHOLDS['pr']:
            status = 'PR_READY'
            threshold = 'pr'
        elif self.score >= THRESHOLDS['commit']:
            status = 'COMMIT_READY'
            threshold = 'commit'
        else:
            status = 'BLOCKED'
            threshold = 'None (below commit)'

        critical_count = len(self.issues['critical'])
        major_count = len(self.issues['major'])
        minor_count = len(self.issues['minor'])
        total_count = critical_count + major_count + minor_count

        return {
            'filepath': str(self.filepath),
            'score': self.score,
            'status': status,
            'threshold': threshold,
            'auto_fail': self.auto_fail,
            'issues': {
                'critical': self.issues['critical'],
                'major': self.issues['major'],
                'minor': self.issues['minor'],
                'counts': {
                    'critical': critical_count,
                    'major': major_count,
                    'minor': minor_count,
                    'total': total_count
                }
            },
            'thresholds': THRESHOLDS
        }

    def print_report(self, summary_only: bool = False) -> None:
        """Print formatted quality report."""
        report = self._generate_report()

        print(f"\n# Quality Score: {self.filepath.name}\n")

        status_label = {
            'EXCELLENCE': '[EXCELLENCE]',
            'PR_READY': '[PASS]',
            'COMMIT_READY': '[PASS]',
            'BLOCKED': '[BLOCKED]',
            'FAIL': '[FAIL]'
        }

        print(f"## Overall Score: {report['score']}/100 {status_label.get(report['status'], '')}")

        if report['status'] == 'BLOCKED':
            print(f"\n**Status:** BLOCKED - Cannot commit (score < {THRESHOLDS['commit']})")
        elif report['status'] == 'COMMIT_READY':
            print(f"\n**Status:** Ready for commit (score >= {THRESHOLDS['commit']})")
            gap_to_pr = THRESHOLDS['pr'] - report['score']
            print(f"**Next milestone:** PR threshold ({THRESHOLDS['pr']}+)")
            print(f"**Gap analysis:** Need +{gap_to_pr} points to reach PR quality")
        elif report['status'] == 'PR_READY':
            print(f"\n**Status:** Ready for PR (score >= {THRESHOLDS['pr']})")
        elif report['status'] == 'EXCELLENCE':
            print(f"\n**Status:** Excellence achieved! (score >= {THRESHOLDS['excellence']})")
        elif report['status'] == 'FAIL':
            print(f"\n**Status:** Auto-fail (compilation/syntax error)")

        if summary_only:
            print(f"\n**Total issues:** {report['issues']['counts']['total']} "
                  f"({report['issues']['counts']['critical']} critical, "
                  f"{report['issues']['counts']['major']} major, "
                  f"{report['issues']['counts']['minor']} minor)")
            return

        print(f"\n## Critical Issues (MUST FIX): {report['issues']['counts']['critical']}")
        if report['issues']['counts']['critical'] == 0:
            print("No critical issues - safe to commit\n")
        else:
            for i, issue in enumerate(report['issues']['critical'], 1):
                print(f"{i}. **{issue['description']}** (-{issue['points']} points)")
                print(f"   - {issue['details']}\n")

        if report['issues']['counts']['major'] > 0:
            print(f"## Major Issues (SHOULD FIX): {report['issues']['counts']['major']}")
            for i, issue in enumerate(report['issues']['major'], 1):
                print(f"{i}. **{issue['description']}** (-{issue['points']} points)")
                print(f"   - {issue['details']}\n")

        if report['issues']['counts']['minor'] > 0 and self.verbose:
            print(f"## Minor Issues (NICE-TO-HAVE): {report['issues']['counts']['minor']}")
            for i, issue in enumerate(report['issues']['minor'], 1):
                print(f"{i}. {issue['description']} (-{issue['points']} points)\n")


# ==============================================================================
# CLI INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Calculate quality scores for TRUST paper materials',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Score the main paper
  python3 scripts/quality_score.py paper/main.tex

  # Score multiple .tex files
  python3 scripts/quality_score.py paper/*.tex

  # Score a Python visualization script
  python3 scripts/quality_score.py scripts/python/plot_results.py

  # Summary only
  python3 scripts/quality_score.py paper/main.tex --summary

Quality Thresholds:
  80/100 = Commit threshold (blocks if below)
  90/100 = PR threshold (warning if below)
  95/100 = Excellence (aspirational)

Exit Codes:
  0 = Score >= 80 (commit allowed)
  1 = Score < 80 (commit blocked)
  2 = Auto-fail (compilation/syntax error)
        """
    )

    parser.add_argument('filepaths', type=Path, nargs='+', help='Path(s) to file(s) to score')
    parser.add_argument('--summary', action='store_true', help='Show summary only')
    parser.add_argument('--verbose', action='store_true', help='Show all issues including minor')
    parser.add_argument('--json', action='store_true', help='Output as JSON')

    args = parser.parse_args()

    results = []
    exit_code = 0

    for filepath in args.filepaths:
        if not filepath.exists():
            print(f"Error: File not found: {filepath}")
            exit_code = 1
            continue

        try:
            scorer = QualityScorer(filepath, verbose=args.verbose)

            if filepath.suffix == '.tex':
                report = scorer.score_paper()
            elif filepath.suffix == '.py':
                report = scorer.score_python()
            else:
                print(f"Error: Unsupported file type: {filepath.suffix} (supported: .tex, .py)")
                continue

            results.append(report)

            if not args.json:
                scorer.print_report(summary_only=args.summary)

            if report['auto_fail']:
                exit_code = max(exit_code, 2)
            elif report['score'] < THRESHOLDS['commit']:
                exit_code = max(exit_code, 1)

        except Exception as e:
            print(f"Error scoring {filepath}: {e}")
            import traceback
            traceback.print_exc()
            exit_code = 1

    if args.json:
        print(json.dumps(results, indent=2))

    sys.exit(exit_code)

if __name__ == '__main__':
    main()
