"""Static sanity check on main.tex, since there is no local LaTeX to compile with."""
import pathlib
import re
from collections import Counter

paper = pathlib.Path("c:/Users/schma/vla-where-does-language-die/paper")
t = (paper / "main.tex").read_text(encoding="utf-8")
# Strip comments, but NOT escaped percents: "$94\%$" is math, not a comment.
# Treating \% as a comment start made the delimiter check report false odd counts.
src = re.sub(r"(?m)(?<!\\)%.*$", "", t)

print("braces balanced       :", src.count("{") == src.count("}"),
      f"({src.count('{')} open / {src.count('}')} close)")
print("$ delimiters even     :", src.count("$") % 2 == 0, f"({src.count('$')})")

begins = Counter(re.findall(r"\\begin\{(\w+\*?)\}", src))
ends = Counter(re.findall(r"\\end\{(\w+\*?)\}", src))
print("environments matched  :", begins == ends,
      dict(begins - ends) or "", dict(ends - begins) or "")

labs = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\ref\{([^}]+)\}", src))
print("undefined \\ref        :", (refs - labs) or "none")

cites = set()
for c in re.findall(r"\\cite[tp]?\{([^}]+)\}", src):
    cites.update(x.strip() for x in c.split(","))
bib = set(re.findall(r"@\w+\{([^,]+),", (paper / "refs.bib").read_text(encoding="utf-8")))
print("cited but not in bib  :", (cites - bib) or "none")
print("in bib but uncited    :", (bib - cites) or "none")

figs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src)
for f in figs:
    print(f"figure {f:28s} exists: {(paper.parent / f).exists()}")

print("has \\workshoptitle    :", "workshoptitle" in src)
print("has \\title            :", bool(re.search(r"\\title\{", src)))
bad = sorted({c for c in t if ord(c) > 127})
print("non-ascii (needs utf8):", "".join(bad) or "none")

# Macros that lost their leading backslash. This happens when text passes through a shell
# heredoc or a naive string replace: "\begin" becomes "egin", "\textbf" becomes "extbf",
# "\ref" becomes a carriage return followed by "ef". LaTeX then renders the fragment as
# literal prose, or fails somewhere unrelated, and the brace/environment counters above
# stay happy because the braces are still balanced. It has bitten this file four times.
STEMS = ["egin", "nd", "extbf", "extit", "mph", "ef", "ection", "aragraph", "abel",
         "itepressure", "itep", "itet", "ubsection", "aption", "includegraphics"]
mangled = []
for lineno, line in enumerate(t.splitlines(), 1):
    for stem in STEMS:
        for _ in re.finditer(r"(?<![A-Za-z\\])" + stem + r"\{", line):
            mangled.append((lineno, stem, line.strip()[:72]))
print("mangled macros        :", len(mangled) or "none")
for lineno, stem, text in mangled[:8]:
    print(f"   line {lineno}: '{stem}{{' -- {text}")

# A lone carriage return mid-line is the same failure caught a different way.
stray_cr = [i for i, line in enumerate(t.split("\n"), 1) if "\r" in line.rstrip("\r")]
print("stray CR mid-line     :", stray_cr or "none")

# Macros whose defining package is not loaded. There is no local LaTeX here, so an
# undefined control sequence only surfaces on Overleaf, hours later. \operatorname and
# \lVert in particular come from amsmath, which amsfonts does NOT pull in.
NEEDS = {
    "amsmath": ["operatorname", "lVert", "rVert", "dfrac"],
    "amssymb": ["mathbb", "leqslant", "geqslant"],
    "booktabs": ["toprule", "midrule", "bottomrule"],
    "graphicx": ["includegraphics"],
    "xcolor": ["textcolor"],
}
groups = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", src)
loaded = {name.strip() for g in groups for name in g.split(",")}
missing = [
    f"\\{macro} needs {pkg}"
    for pkg, macros in NEEDS.items()
    if pkg not in loaded
    for macro in macros
    if re.search(r"\\" + macro + r"[\{\[ ]", src)
]
print("missing packages      :", "; ".join(sorted(set(missing))) or "none")

# Table rows that lost a backslash. Same collapse as the mangled macros above, but it hits
# the "\\" row terminator, turning it into a line-continuation "\". The braces stay
# balanced and no macro name is damaged, so every other check here passes while the table
# silently runs all its rows together.
bad_rows = []
in_tab = False
for lineno, line in enumerate(src.splitlines(), 1):
    stripped = line.strip()
    if re.match(r"\\begin\{(tabular|array)", stripped):
        in_tab = True
        continue
    if re.match(r"\\end\{(tabular|array)", stripped):
        in_tab = False
        continue
    if not in_tab or not stripped:
        continue
    if re.match(r"\\(top|mid|bottom|cmid)rule|\\hline|\\multicolumn", stripped):
        continue
    if "&" in stripped and not stripped.endswith(r"\\"):
        bad_rows.append((lineno, stripped[:64]))
print("table rows missing \\\\ :", len(bad_rows) or "none")
for lineno, text in bad_rows[:8]:
    print(f"   line {lineno}: {text}")
