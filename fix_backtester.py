#!/usr/bin/env python3
"""Quick script to fix the backtester syntax errors"""

# Read the file
with open('jeafx_backtester.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the specific issue with escaped newlines in dataclass strings
lines = content.split('\n')
fixed_lines = []

for line in lines:
    # If line contains escaped newline markers, fix them
    if '\\n' in line and ('"""' in line or '@dataclass' in line or 'class ' in line):
        # Remove escaped newlines and fix formatting
        fixed_line = line.replace('\\n    ', '\n    ').replace('\\n', '\n')
        if fixed_line.count('\n') > 1:
            # Split multi-line content and add each as separate lines
            split_content = fixed_line.split('\n')
            fixed_lines.extend(split_content)
        else:
            fixed_lines.append(fixed_line)
    else:
        fixed_lines.append(line)

# Write the fixed content
with open('jeafx_backtester.py', 'w', encoding='utf-8') as f:
    f.write('\n'.join(fixed_lines))

print("✅ Fixed jeafx_backtester.py syntax errors")