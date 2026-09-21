# arXiv submission package

Source for the paper describing `jepctl`. Written against the arXiv
[submission guidelines](https://info.arxiv.org/help/submit/index.html) and
[format requirements](https://info.arxiv.org/help/policies/format_requirements.html).

## Files

| File | Purpose |
|---|---|
| `jepctl.tex` | The paper. Standard `article` class, no exotic packages. |
| `refs.bib` | BibTeX database. arXiv accepts `.bib` **or** `.bbl`; see below. |

## Build

No TeX distribution is installed in the repository, so this source has **not been
compiled here**. Build it with either:

```bash
# MacTeX / TeX Live
pdflatex jepctl && bibtex jepctl && pdflatex jepctl && pdflatex jepctl

# or a single self-contained binary
brew install tectonic && tectonic jepctl.tex
```

Or paste `jepctl.tex` and `refs.bib` into a new [Overleaf](https://overleaf.com) project,
which compiles both and lets you download the `.bbl`.

## Before you submit

1. **Complete the author block.** `jepctl.tex` carries two `% TODO(author)` comments:
   the full name and affiliation on the `\author` line, and the acknowledgements section.
   arXiv reproduces the author block verbatim on the abstract page.
2. **Ship the `.bbl`.** arXiv runs BibTeX only if you include `refs.bib`, and the safest
   upload is `jepctl.tex` **plus the generated `jepctl.bbl`** (produced by the build
   above). Include both and the build is deterministic on their side.
3. **Pick the subject class.** `cs.LG` (Machine Learning) fits this paper best, with
   `cs.CV` and `cs.RO` as cross-lists. The two cited works sit in `cs.LG`
   (arXiv:2603.19312) and `cs.CL` (arXiv:2609.20800).
4. **Pick a licence.** arXiv requires an irrevocable distribution licence and the choice
   **cannot be changed after submission**. CC BY 4.0 is the usual choice for a software
   paper whose code is already Apache-2.0; check any target venue's preprint policy first.
5. **Check the public links resolve.** The paper links to
   `https://github.com/younss/jepctl`. arXiv requires code and data links to resolve to a
   publicly available repository, so make sure the repository is public before submitting.

## Format compliance

The source already satisfies the stated requirements: no line numbers, single-spaced,
11 pt type, 1-inch margins on every side, no watermark, no highlighted text, no embedded
JavaScript, and no animated figures. Both figures are TikZ, so the PDF contains vector
graphics and no raster assets.

## A note on the image you suggested

The image at `miro.medium.com/.../1*Ee9SwJ-8L250WIQtCr-CJQ.jpeg` is hosted on Medium and
is almost certainly covered by someone else's copyright. arXiv requires the submitter to
hold the rights to everything in the submission, so including it would put the submission
at risk and could get it removed. I did not include it.

Instead the paper carries two original figures drawn in TikZ:

- **Figure 1** — the `jepctl` architecture: sensors, preprocessing, encoder, and the three
  latent-space consumers (few-shot matcher, online world model, control).
- **Figure 2** — the World loop: frozen encoder, online predictor, surprise as the
  normalised prediction error, following the structure in arXiv:2603.19312.

If you want the picture you linked, the clean routes are: (a) reuse the original figure
from the paper it came from and cite it, only if that paper's licence allows reuse (arXiv
papers under CC BY do; "arXiv.org perpetual licence" ones do not), (b) ask the Medium
author for written permission, or (c) ask me to redraw the same concept as an original
TikZ figure, which is free of any rights question. I would recommend (c).
