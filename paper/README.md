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

## Figures

- **Figure 1** — `figures/jepa-overview.jpg`, the author's own overview infographic
  (1400x764 JPEG, downloaded from the author's blog). Since it is your own work there is
  no third-party rights question; arXiv only requires that you hold the rights.
  Two things worth deciding before you submit:
  1. The image was generated with an assistant and contains a few garbled strings (for
     example "Train to trinring" near the predictor, and the mock terminal text). Reviewers
     do notice this. Regenerating it with the text corrected, or overlaying clean labels,
     would make the paper look tighter.
  2. Some venues now ask authors to disclose AI-generated figures. arXiv does not currently
     require it for figures, but adding "generated with the assistance of an image model"
     to the caption costs nothing and pre-empts the question.
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

