#!/usr/bin/env python3
"""Lexical fast gate; Audit.lean independently inspects transitive kernel dependencies."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / 'lean'


def strip_comments_strings(text):
    # Lean supports nested block comments. Preserve whitespace boundaries.
    out, at, depth = [], 0, 0
    while at < len(text):
        if text.startswith('/-', at):
            depth += 1; out.append(' '); at += 2
        elif depth and text.startswith('-/', at):
            depth -= 1; at += 2
        elif depth:
            at += 1
        elif text.startswith('--', at):
            end = text.find('\n', at)
            at = len(text) if end < 0 else end
        elif text[at] == '"':
            out.append(' '); at += 1
            while at < len(text):
                if text[at] == '\\': at += 2
                elif text[at] == '"': at += 1; break
                else: at += 1
        else:
            out.append(text[at]); at += 1
    if depth: raise ValueError('unterminated Lean comment')
    return ''.join(out)


def check_source(text):
    code = strip_comments_strings(text)
    # Escaped identifiers and Unicode spellings still face the kernel axiom audit.
    if re.search(r'\b(sorry|admit|axiom|axioms|unsafe)\b', code):
        # #print axioms is a diagnostic, not an axiom declaration.
        code = re.sub(r'#print\s+axioms\b', '', code)
        code = re.sub(r'\.axioms\b', '', code)
        if re.search(r'\b(sorry|admit|axiom|axioms|unsafe)\b', code):
            raise ValueError('unfinished proof or unapproved declaration')
    if re.search(r'\b(set_option\s+(debug|Elab\.async)|run_tac)\b', code):
        raise ValueError('unsupported proof escape')

if __name__ == '__main__':
    files = [*sorted(ROOT.glob('*.lean')), *sorted((ROOT / 'Pudding').rglob('*.lean'))]
    for file in files:
        try: check_source(file.read_text())
        except ValueError as e: raise SystemExit(f'{file}: {e}')
    print(f'{len(files)} Lean sources pass the no-placeholder gate')
