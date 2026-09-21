# arXiv submission package

Source for the paper describing `jepctl`. Written against the arXiv
[submission guidelines](https://info.arxiv.org/help/submit/index.html) and
[format requirements](https://info.arxiv.org/help/policies/format_requirements.html).

## Files

| File | Purpose |
|---|---|
| `jepctl.tex` | The paper, written as an engineering report: it documents a tool and states explicitly, in Section 1.1, that it makes no research claim. Standard `article` class, no exotic packages. |
| `refs.bib` | BibTeX database. arXiv accepts `.bib` **or** `.bbl`; see below. |

## Build

The PDF in this directory (`jepctl.pdf`, 10 pages) was produced from this source with
Tectonic 0.17.0 and compiles with **zero warnings**. `jepctl.bbl` is the generated
bibliography and is tracked on purpose, because arXiv wants it in the upload. To rebuild:

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
2. **Ship the `.bbl`.** The safest upload is `jepctl.tex` plus `jepctl.bbl` plus
   `figures/jepa-overview.jpg`; both are already in this directory, so the build is
   deterministic on arXiv's side. (`refs.bib` is kept here for future edits.)
3. **Pick the subject class.** This is a tools/engineering report, so **`cs.SE`
   (Software Engineering)** is the honest primary category, with `cs.LG` and `cs.CV` as
   cross-lists (`cs.RO` too if you want the robot audience). Submitting a tool paper to
   `cs.LG` as primary invites reviewers to judge it as a research contribution, which it
   does not claim to be. The two cited works sit in `cs.LG` (arXiv:2603.19312) and
   `cs.CL` (arXiv:2609.20800).
4. **Pick a licence.** arXiv requires an irrevocable distribution licence and the choice
   **cannot be changed after submission**. CC BY 4.0 is the usual choice for a software
   paper whose code is already Apache-2.0; check any target venue's preprint policy first.
5. **Check the public links resolve.** The paper links to
   `https://github.com/younss/jepctl`. arXiv requires code and data links to resolve to a
   publicly available repository, so make sure the repository is public before submitting.

## Format compliance

The source already satisfies the stated requirements: no line numbers, single-spaced,
11 pt type, 1-inch margins on every side, no watermark, no highlighted text, no embedded
JavaScript, and no animated figures. Figures 2 and 3 are TikZ vector graphics; Figure 1 is
a single static JPEG.

## Figures

- **Figure 1** — `figures/jepa-overview.jpg`, the author's own overview infographic
  (1400x764 JPEG, downloaded from the author's blog). Since it is your own work there is
  no third-party rights question; arXiv only requires that you hold the rights.
  The artwork carried three garbled strings from the image generator; they are covered in
  `jepctl.tex` by a TikZ overlay drawn on top of the image, so the text is vector, correct
  and legible while the artwork itself is untouched:
  1. the mock terminal, repainted with real `jepctl serve` output;
  2. `vo:` corrected to `via:` after "API control examples";
  3. `Train to tcrinring` replaced by "Fit on observed transitions".
  The overlay uses normalised coordinates inside a `scope` keyed to the image node, so if
  you ever regenerate the artwork you only need to re-measure those three rectangles.
  Note: some venues ask authors to disclose AI-generated figures. arXiv does not currently
  require it, but a line in the caption costs nothing and pre-empts the question.
- **Figure 2** — TikZ, the `jepctl` dataflow: sensors, preprocessing, encoder, and the
  three latent-space consumers (few-shot matcher, online world model, control).
- **Figure 3** — TikZ, the World loop: frozen encoder, online predictor, surprise as the
  normalised prediction error, following the structure in arXiv:2603.19312.

Figures 2 and 3 are vector graphics; Figure 1 is the only raster asset, at 198 KB, well
within arXiv's limits.

## Reference provenance

Every arXiv reference was checked against its abstract page during preparation, not
written from memory:

| Key | Identifier | Verified |
|---|---|---|
| `lewm2026` | arXiv:2603.19312 | full PDF read |
| `jepaanything2026` | arXiv:2609.20800 | abstract page |
| `ijepa2023` | arXiv:2301.08243 | abstract page |
| `vjepa2_2025` | arXiv:2506.09985 | abstract page (first author is **Mido** Assran) |
| `dinov2_2023` | arXiv:2304.07193 | abstract page |
| `vit2021` | arXiv:2010.11929 | abstract page |
| `audiomae2022` | arXiv:2207.06405 | abstract page (NeurIPS 2022) |
| `lecun2022path` | OpenReview `BZ5a1r-kVsf` | confirmed by search; page itself is behind a bot check |
| `cem2004` | Rubinstein & Kroese, Springer 2004 | matches the citation in arXiv:2603.19312's own bibliography |
| `jl1984` | Johnson & Lindenstrauss, Contemp. Math. 26 | standard citation, **not** re-verified online |
| `candle`, `safetensors`, `ollama` | GitHub URLs | project dependencies / well-known repositories |

