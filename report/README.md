# Report

`report.tex` — IEEE conference template, 2-column, 10pt. Main text ends on page 4;
page 5 is references, which the guidelines exclude from the limit.

## Before submitting

Fill the two placeholders on lines 31–32 of `report.tex`:

```
Student ID: <<<INSERT STUDENT ID>>>
Email:      <<<INSERT UNITN EMAIL>>>
```

Requirement 1 of the guidelines is Name, Surname, ID, email — a report missing
any requirement is not evaluated. Also confirm the author name is spelled as you
want it; it was taken from the cluster account and not verified.

## Build

```bash
pdflatex report.tex && pdflatex report.tex     # twice, for cross-references
```

Needs `IEEEtran.cls`, `algorithm`, `algpseudocode`, `booktabs`. On Debian/Ubuntu:
`texlive-publishers texlive-science texlive-latex-extra`.

Figures are pulled from `../plots/` and regenerate from the committed CSVs, so
the report tracks the data automatically.

## Checking the page limit after editing

`\label{endofcontent}` sits immediately before the bibliography. After a rebuild:

```bash
grep endofcontent report.aux      # the second {...} is the page number; must be <= 4
```

## When the cluster campaign lands

The reported results are single-node and the text says so in three places —
Experiments (measurement platform), Results (the strong-scaling paragraph) and
Conclusions (third limitation). When multi-node CSVs arrive:

1. regenerate figures from the cluster CSVs;
2. update the numbers in Results;
3. soften those three caveats to match what the data then supports;
4. re-check the page count as above.
