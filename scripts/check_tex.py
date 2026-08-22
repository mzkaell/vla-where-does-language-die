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
