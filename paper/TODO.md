# LaTeX paper (arXiv preprint, in preparation)

`main.tex` is scaffolded with section headings, abstract, tables, and figure
references. Section bodies need to be filled in from the README narrative.

## Build

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with latexmk:

```bash
latexmk -pdf main.tex
```

## Target

arXiv category: **q-fin.ST** (Statistical Finance) primary, **stat.AP** secondary.

Bentley fintech affiliation should suffice; if not, request endorsement from
a faculty member in the Mathematical Sciences department.

## Status

- [x] Scaffold sections + tables + figure references
- [x] BibTeX seed entries for data sources
- [ ] Fill Introduction (port from README §Background)
- [ ] Fill Background §§
- [ ] Fill Architecture §§ (port from README architecture diagram)
- [ ] Fill Results §§ (port from README headline + correlation diagnostic)
- [ ] Fill Mechanism § (port CLT argument from README)
- [ ] Fill Generalization § (port from README §implications)
- [ ] Fill Reproducibility § (point to repo)
- [ ] Fill Conclusion §
- [ ] Add CSW% citation, deep-research synthesis citations
- [ ] Polish abstract
- [ ] Generate PDF and review
