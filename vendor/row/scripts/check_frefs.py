#!/usr/bin/env python3
"""F-reference gate: every F-x.y (and F-\\S x.y) token in the paper
must correspond to a number that exists in the foundations paper's
regenerated .aux (theorem-counter results, sections, or equations).

Usage: check_frefs.py paper.tex foundations.aux
Exit 1 on any unmatched token, listing it with its line number.
"""

import re
import sys


def aux_numbers(aux_path):
    """All printed numbers defined by \\newlabel entries: {number}."""
    nums = set()
    pat = re.compile(r"\\newlabel\{[^}]*\}\{\{([^{}]*)\}")
    for line in open(aux_path, errors="replace"):
        m = pat.search(line)
        if m:
            nums.add(m.group(1))
    return nums


def paper_tokens(tex_path):
    """(lineno, kind, number) for every F- token; kind is 'sec' for
    F-\\S references and 'num' otherwise."""
    out = []
    tok = re.compile(r"F-(\\S)?([0-9]+(?:\.[0-9]+)*)")
    for i, line in enumerate(open(tex_path, errors="replace"), 1):
        if line.lstrip().startswith("%"):
            continue
        for m in tok.finditer(line):
            out.append((i, "sec" if m.group(1) else "num", m.group(2)))
    return out


def main():
    tex, aux = sys.argv[1], sys.argv[2]
    nums = aux_numbers(aux)
    bad = []
    for lineno, kind, number in paper_tokens(tex):
        if number not in nums:
            bad.append((lineno, kind, number))
    if bad:
        for lineno, kind, number in bad:
            print(f"UNMATCHED F-ref at line {lineno}: "
                  f"{'section ' if kind == 'sec' else ''}{number}")
        sys.exit(1)
    n = len(paper_tokens(tex))
    print(f"check_frefs: {n} F-reference tokens verified against "
          f"{len(nums)} aux numbers")


if __name__ == "__main__":
    main()
