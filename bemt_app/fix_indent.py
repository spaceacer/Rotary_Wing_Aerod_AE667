import re

with open('pages/2_BEMT_Dashboard.py', 'r') as f:
    lines = f.readlines()

new_lines = []
in_fragment = False

for line in lines:
    if line.startswith('@st.fragment'):
        in_fragment = True
        new_lines.append(line)
    elif line.startswith('with tab_inspect:'):
        in_fragment = False
        new_lines.append(line)
    elif in_fragment:
        if line.startswith('        '):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('pages/2_BEMT_Dashboard.py', 'w') as f:
    f.writelines(new_lines)
