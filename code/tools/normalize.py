#!/usr/bin/env python3
"""Input normalization rules N1 and N2 used in the normalization experiments.

N1: convert CRLF/CR to LF, strip trailing spaces and tabs from every line,
    drop lines that are empty after stripping (so no blank lines remain).
N2: N1 followed by lower-casing.

Usage: normalize.py {N1,N2} < input.txt > output.txt
"""
import sys


def normalize(text, rule):
    if rule not in ('N1', 'N2'):
        raise ValueError(rule)
    lines = [line.rstrip(' \t') for line in text.replace('\r\n', '\n').replace('\r', '\n').split('\n')]
    out = '\n'.join(line for line in lines if line != '')
    return out.lower() if rule == 'N2' else out


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('N1', 'N2'):
        sys.exit(__doc__.strip())
    sys.stdout.write(normalize(sys.stdin.read(), sys.argv[1]))
