import re

with open('pages/2_BEMT_Dashboard.py', 'r') as f:
    content = f.read()

split_marker = 'with tab_inspect:\n    st.subheader("Spanwise Station Inspector")'
parts = content.split(split_marker)

if len(parts) != 2:
    print("Could not find tab_inspect marker")
    exit(1)

head = parts[0]
tail = parts[1]

fragment_def = """
@st.fragment
def render_station_inspector(af_blend, res, geom, cond, r_norm, deg_alpha, r_rc, radius):
    st.subheader("Spanwise Station Inspector")
"""

indented_tail = "\n".join(["    " + line if line else line for line in tail.split("\n")])

new_tail = fragment_def + indented_tail + """
with tab_inspect:
    render_station_inspector(af_blend, res, geom, cond, r_norm, deg_alpha, r_rc, radius)
"""

new_content = head + new_tail

with open('pages/2_BEMT_Dashboard.py', 'w') as f:
    f.write(new_content)
