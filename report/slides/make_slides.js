// Generates report/slides/presentation.pptx — 15-minute project presentation.
// Run: node report/slides/make_slides.js   (from the repo root)
const pptxgen = require('pptxgenjs');
const path = require('path');
const ROOT = path.resolve(__dirname, '..', '..');
const plot = (f) => path.join(ROOT, 'plots', f);

const pres = new pptxgen();
pres.layout = 'LAYOUT_16x9'; // 10" x 5.625"
pres.author = 'Saif Edine Safi';
pres.title = 'A Generic Distributed-Memory GEMM with a Cost-Model-Driven Decomposition Planner';

const NAVY = '1E2761', INK = '222222', MUTED = '666666', ACCENT = 'C8102E', LIGHT = 'F2F4F8', WHITE = 'FFFFFF';
const FONT = 'Calibri';

function title(slide, text, sub) {
  slide.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.7, fontFace: FONT, fontSize: 24, bold: true, color: NAVY, valign: 'top', isTextBox: true, margin: 0 });
  if (sub) slide.addText(sub, { x: 0.5, y: 0.95, w: 9, h: 0.4, fontFace: FONT, fontSize: 14, italic: true, color: MUTED, isTextBox: true, margin: 0 });
}
function bullets(slide, items, opts = {}) {
  const runs = items.map((t, i) => {
    const o = typeof t === 'string' ? { text: t } : t;
    return { text: o.text, options: { bullet: o.sub ? { indent: 15 } : true, indentLevel: o.sub ? 1 : 0, fontSize: o.sub ? (opts.fontSize || 15) - 1.5 : (opts.fontSize || 15), bold: !!o.bold, color: INK, breakLine: i < items.length - 1, paraSpaceAfter: 6 } };
  });
  slide.addText(runs, { x: opts.x ?? 0.5, y: opts.y ?? 1.45, w: opts.w ?? 9, h: opts.h ?? 3.7, fontFace: FONT, valign: 'top', isTextBox: true, margin: 0 });
}
function stat(slide, x, y, w, big, label) {
  slide.addShape(pres.ShapeType.roundRect, { x, y, w, h: 1.25, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  slide.addText(big, { x, y: y + 0.1, w, h: 0.65, fontFace: FONT, fontSize: 30, bold: true, color: ACCENT, align: 'center', isTextBox: true, margin: 0 });
  slide.addText(label, { x: x + 0.1, y: y + 0.75, w: w - 0.2, h: 0.45, fontFace: FONT, fontSize: 11, color: MUTED, align: 'center', isTextBox: true, margin: 0 });
}
function footer(slide, n) {
  slide.addText(`gemm2d — ${n}`, { x: 8.3, y: 5.25, w: 1.5, h: 0.25, fontFace: FONT, fontSize: 9, color: MUTED, align: 'right', isTextBox: true, margin: 0 });
}
function notes(slide, text) { slide.addNotes(text.trim()); }

let n = 0;

// ---------------------------------------------------------------- 1. Title
{
  const s = pres.addSlide(); n++;
  s.background = { color: NAVY };
  s.addText('A Generic Distributed-Memory GEMM\nwith a Cost-Model-Driven Decomposition Planner', { x: 0.6, y: 1.3, w: 8.8, h: 1.6, fontFace: FONT, fontSize: 30, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText('gemm2d  ·  C ← αAB + βC over a Pr × Pc × c process grid', { x: 0.6, y: 3.0, w: 8.8, h: 0.5, fontFace: FONT, fontSize: 16, color: 'CADCFC', isTextBox: true, margin: 0 });
  s.addText('Saif Edine Safi  ·  Student ID 245473\nParallel Computing — final project  ·  University of Trento', { x: 0.6, y: 4.1, w: 8.8, h: 0.8, fontFace: FONT, fontSize: 13, color: 'CADCFC', isTextBox: true, margin: 0 });
  notes(s, `
[~0:30] Good morning/afternoon. My project is "gemm2d": a distributed-memory matrix multiplication — GEMM — that is *generic*: it works for any matrix shape and any number of processes, and it chooses its own decomposition at run time with a cost model.

GEMM is C = alpha*A*B + beta*C. A is M by K, B is K by N, C is M by N. The matrices are spread over P MPI processes arranged in a 3D grid Pr x Pc x c.

Plan for the next 15 minutes: the problem, the design (layout, engines, planner, kernel, verification), then five results and the honest limitations.
`);
}

// ---------------------------------------------------------------- 2. The problem
{
  const s = pres.addSlide(); n++;
  title(s, 'The problem: “generic” is the hard part', 'Demonstration GEMMs make three assumptions. All three fail in practice.');
  const cards = [
    ['M = N = K', 'Real workloads are skewed. A Gram matrix AᵀA has K ≫ M, N. A deep-learning layer often has K ≪ M, N.'],
    ['P is a perfect square', 'The scheduler grants whatever it grants — including prime process counts like 7 or 13.'],
    ['√P divides n', 'Dimensions almost never divide evenly. Padding wastes flops and hides bugs.'],
  ];
  cards.forEach(([h, b], i) => {
    const x = 0.5 + i * 3.05;
    s.addShape(pres.ShapeType.roundRect, { x, y: 1.55, w: 2.9, h: 2.4, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
    s.addText('✗  ' + h, { x: x + 0.2, y: 1.7, w: 2.5, h: 0.5, fontFace: FONT, fontSize: 16, bold: true, color: ACCENT, isTextBox: true, margin: 0 });
    s.addText(b, { x: x + 0.2, y: 2.25, w: 2.5, h: 1.6, fontFace: FONT, fontSize: 13, color: INK, valign: 'top', isTextBox: true, margin: 0 });
  });
  s.addText('In distributed memory, GEMM cost is dominated by data movement, not arithmetic — and data movement is decided by how operands are laid out across processes.', { x: 0.5, y: 4.2, w: 9, h: 0.8, fontFace: FONT, fontSize: 14, italic: true, color: NAVY, isTextBox: true, margin: 0 });
  footer(s, n);
  notes(s, `
[~1:00] Why is this a project at all? Matrix multiply is the kernel that dense linear algebra and most of deep learning reduce to. On one machine it's compute-bound; across many machines it's dominated by *data movement*, and how much data moves is decided by the layout of the operands across processes.

The word "generic" in the project title is the whole problem. A textbook implementation assumes three things: square matrices, a perfect-square process count, and dimensions divisible by the grid. Each fails in practice:
- Shapes are skewed. A Gram matrix A-transpose-A has a huge K. A neural network layer often has a tiny K.
- The scheduler gives you whatever it gives you — 7 processes, 13 processes.
- Dimensions don't divide. The usual fix — zero padding — wastes work and, worse, hides bugs, because zeros make wrong results look plausible.

So the question is: can we drop all three assumptions without giving up performance?
`);
}

// ---------------------------------------------------------------- 3. Goals & scope
{
  const s = pres.addSlide(); n++;
  title(s, 'Goals and scope');
  s.addText('Genericity on three axes', { x: 0.5, y: 1.4, w: 4.4, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    { text: 'G1 — shape: arbitrary M, N, K with α, β scaling  ✓', bold: true },
    { text: 'G2 — machine: arbitrary P (incl. primes), arbitrary grids, non-divisible dimensions  ✓', bold: true },
    { text: 'G3 — type: float / double through one code path  (scoped out)' },
  ], { x: 0.5, y: 1.85, w: 4.4, h: 2.2, fontSize: 14 });
  s.addText('Four objectives', { x: 5.2, y: 1.4, w: 4.3, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    'Correct for arbitrary M, N, K, P',
    'A local kernel fast enough that scaling measures the decomposition, not the inner loop',
    'A run-time planner that picks the grid from a calibrated model, not by convention',
    'A verification strategy usable at full problem size',
  ], { x: 5.2, y: 1.85, w: 4.3, h: 2.4, fontSize: 14 });
  s.addText('Explicit non-goals: sparse matrices, Strassen-class algorithms, GPU offload, fault tolerance.', { x: 0.5, y: 4.55, w: 9, h: 0.4, fontFace: FONT, fontSize: 12, color: MUTED, isTextBox: true, margin: 0 });
  footer(s, n);
  notes(s, `
[~0:45] I define genericity on three axes. G1, shape: any M, N, K, plus the alpha/beta scaling. G2, machine: any process count including primes, any grid, non-divisible dimensions. Both are implemented and tested. G3, scalar type, I scoped out — the code is double-only, and I'll come back to it under future work.

Four objectives. One: correctness for arbitrary inputs. Two: a fast local kernel — if the inner loop is slow, every scaling curve just measures the inner loop and the communication behaviour is invisible. Three: a planner that chooses the process grid at run time from a calibrated cost model. Four: verification that still works at full size, where you can't compare against a sequential reference.

Non-goals, so nobody expects them: sparse, Strassen, GPU, fault tolerance.
`);
}

// ---------------------------------------------------------------- 4. State of the art
{
  const s = pres.addSlide(); n++;
  title(s, 'State of the art — and the gap');
  const rows = [
    [{ text: 'Method', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'Idea', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'Limitation', options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ['Cannon (1969)', 'Skew + √P nearest-neighbour shifts; optimal bandwidth, O(√P) messages', 'Square grid, divisible dims only — exactly what G2 rejects'],
    ['SUMMA (1997)', 'Broadcast b-wide panels along process rows/columns', 'Natively generic in shape and grid → our base engine'],
    ['ScaLAPACK', '2D block-cyclic layout', 'Block-cyclic exists for LU/QR imbalance; GEMM work is uniform → we use plain 2D blocks'],
    ['2.5D (Solomonik & Demmel 2011)', 'Replicate over c layers, split k; bandwidth ÷ √c', 'Reduction cost grows with c; optimum near c ≈ P^(1/3)'],
    ['COSMA (2019)', 'Near I/O-optimal decomposition for any M, N, K, P, memory', 'Elaborate optimality argument; engineering default is still a hard-coded √P × √P grid'],
  ];
  s.addTable(rows, { x: 0.5, y: 1.35, w: 9, colW: [1.9, 3.6, 3.5], fontFace: FONT, fontSize: 10, color: INK, border: { type: 'solid', pt: 0.5, color: 'CCCCCC' }, valign: 'middle', rowH: 0.4 });
  s.addText('Our gap: how much of COSMA’s benefit is recovered by simply enumerating the grid factorisations of P and minimising a calibrated cost model? A few dozen lines, droppable into any SUMMA.', { x: 0.5, y: 4.55, w: 9, h: 0.6, fontFace: FONT, fontSize: 12, bold: true, color: ACCENT, isTextBox: true, margin: 0 });
  footer(s, n);
  notes(s, `
[~1:15] Quick tour of the literature, because it motivates the design choices.

Cannon's algorithm, 1969, is the classic 2D method: skew the matrices, then do square-root-of-P nearest-neighbour shifts. Bandwidth-optimal with very few messages, but it only works on a square grid with divisible dimensions — precisely the assumptions we're rejecting.

SUMMA, 1997, replaces the shifts with broadcasts of narrow panels along process rows and columns. The panel width b is decoupled from the grid, so SUMMA is naturally generic. That's why it's our base engine.

ScaLAPACK uses a block-cyclic layout. Block-cyclic exists to balance the triangular work in LU and QR; GEMM's work is uniform, so I deliberately use a plain 2D block distribution and avoid the index arithmetic.

Communication lower bounds show that 2D algorithms are optimal only when memory is tight. With more memory, replication reduces communication — that's the 2.5D idea: split the k dimension over c layers, bandwidth drops by root-c, at the cost of a reduction at the end. The optimum is around c equals P to the one-third.

COSMA is the state of the art: a near I/O-optimal decomposition for any combination of dimensions, process count and memory. Its optimality argument is elaborate. Meanwhile the common engineering default is still a hard-coded square grid regardless of shape.

The gap I address is narrower and practical: how much of COSMA's benefit do you get by enumerating the small space of grid factorisations and minimising a *calibrated* performance model? It's a cost-model selector, explicitly not an I/O-optimality proof.
`);
}

// ---------------------------------------------------------------- 5. Architecture / data distribution
{
  const s = pres.addSlide(); n++;
  title(s, 'One layout, four engines', 'Every engine shares a single distributed-matrix descriptor');
  bullets(s, [
    { text: 'Descriptor: global dims, local block dims + leading dimension, global index of local (0,0)', bold: true },
    { text: 'Balanced uneven blocks, no padding:  size(i) = ⌊G/P⌋ + [ i < G mod P ]', bold: true },
    { text: 'Padding would waste O((Pr+Pc)·n) flops and hide bugs', sub: true },
    { text: 'Ragged last blocks, 1×P grids and P > M all reduce to this one partition function', sub: true },
    { text: 'No scatter from rank 0: each process generates its block from a deterministic index hash', bold: true },
    { text: 'Element (i, j) has the same value for any P and any grid → results bit-comparable across configurations', sub: true },
    { text: 'Row / column / depth communicators via MPI_Comm_split on a Pr × Pc × c grid', bold: true },
  ], { x: 0.5, y: 1.45, w: 5.6, h: 3.6, fontSize: 13 });
  // engine cards
  const eng = [['naive1d', 'baseline; per-rank volume K·N independent of P'], ['cannon', 'baseline; refuses non-square / non-divisible'], ['summa', 'base engine; generic panels'], ['summa25d', 'SUMMA per k-slab + MPI_Reduce over depth']];
  eng.forEach(([name, d], i) => {
    const y = 1.5 + i * 0.9;
    s.addShape(pres.ShapeType.roundRect, { x: 6.4, y, w: 3.1, h: 0.78, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
    s.addText(name, { x: 6.55, y: y + 0.06, w: 2.8, h: 0.3, fontFace: 'Courier New', fontSize: 13, bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(d, { x: 6.55, y: y + 0.36, w: 2.8, h: 0.4, fontFace: FONT, fontSize: 10, color: MUTED, isTextBox: true, margin: 0 });
  });
  footer(s, n);
  notes(s, `
[~1:15] Now the design. All four engines share one data layout. A distributed matrix is a descriptor: the global dimensions, the local block dimensions and leading dimension, and the *global* index of the local element (0,0).

Non-divisible dimensions are handled by balanced uneven blocks, not by zero padding: part i of G items over P parts gets floor(G/P) plus one if i is less than G mod P. I'd call concentrating every edge case into this one partition function the single most valuable structural decision in the code: ragged last blocks, degenerate 1-by-P grids and the case where there are more processes than rows all reduce to it.

Second point: operands are never built on rank 0 and scattered. Each process generates its own block from a deterministic hash of the global index. This removes a memory bottleneck and, importantly, means element (i,j) has the same value for any P and any grid — so results are bit-comparable across configurations, which the tests rely on.

Processes form a Pr by Pc by c grid, with row, column and depth communicators from MPI_Comm_split.

The four engines, on the right: naive 1D and Cannon are baselines — Cannon *refuses* non-square grids and non-divisible dimensions rather than silently padding; that refusal is the measured argument for choosing SUMMA as the base. SUMMA is the main engine, and 2.5D SUMMA adds replication over depth.
`);
}

// ---------------------------------------------------------------- 6. Generic SUMMA + 2.5D
{
  const s = pres.addSlide(); n++;
  title(s, 'Generic SUMMA and the 2.5D engine');
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.35, w: 4.6, h: 3.7, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  s.addText('Generic SUMMA over k-range [K₀, K₀+Kℓ)', { x: 0.7, y: 1.45, w: 4.3, h: 0.35, fontFace: FONT, fontSize: 13, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  s.addText([
    'scale local C by β  (once)', 'k ← 0', 'while k < Kℓ:', '   r_c ← Owner(k, Pc);  r_r ← Owner(k, Pr)', '   k_b ← min(k+b, End(r_c), End(r_r), Kℓ) − k', '   if my_col = r_c: pack A(:, k:k+k_b)', '   Bcast(A_buf, m·k_b, r_c, row_comm)', '   if my_row = r_r: pack B(k:k+k_b, :)', '   Bcast(B_buf, k_b·n, r_r, col_comm)', '   C += α · A_buf · B_buf', '   k ← k + k_b',
  ].map((t, i, a) => ({ text: t, options: { breakLine: i < a.length - 1 } })), { x: 0.7, y: 1.85, w: 4.3, h: 3.1, fontFace: 'Courier New', fontSize: 10.5, color: INK, valign: 'top', isTextBox: true, margin: 0 });
  bullets(s, [
    { text: 'The generic subtlety: A’s k-dimension is split over Pc, B’s over Pr — the two partitions differ', bold: true },
    { text: 'A panel of width b may straddle an ownership boundary in either operand → each step is clipped to the intersection of both owners’ ranges', sub: true },
    { text: '2.5D: layer ℓ gets k-slab [Off(K,c,ℓ), +Size(K,c,ℓ)), runs a full SUMMA, then one MPI_Reduce over depth', bold: true },
    { text: 'k-slab variant (not the original A/B replication) composes with SUMMA in ~40 lines; only extra memory is c copies of C', sub: true },
    { text: 'βC applied on layer 0 only, other layers start from zero → αAB + βC exactly once', sub: true },
  ], { x: 5.4, y: 1.4, w: 4.1, h: 3.7, fontSize: 12 });
  footer(s, n);
  notes(s, `
[~1:15] Here is the SUMMA loop, and the one subtlety that makes it generic.

SUMMA walks along k in panels of width b. At each step, the process column owning that slice of A broadcasts it along the process row; the process row owning that slice of B broadcasts it down the process column; then everyone does a local rank-b update of C.

The subtlety: A's k-dimension is partitioned over Pc columns, but B's k-dimension is partitioned over Pr rows. Those two partitions are different — unless the grid is square and everything divides. So a panel of width b may cross an ownership boundary in A, or in B, or both. The fix is line 5: each step is clipped to the intersection of both owners' ranges. That's what lets any grid and any dimension work, and the clipping is exercised by panel widths from 1 to 4096 in the test suite.

The 2.5D engine on the right: each of the c layers gets a contiguous slab of the k dimension, runs a complete SUMMA on that slab, and then the partial C's are summed with one MPI_Reduce over the depth communicator. I used the k-slab formulation rather than Solomonik and Demmel's original, which replicates A and B across layers: the k-slab version reuses the SUMMA engine unchanged in about forty lines, and its only extra memory is c copies of C. Beta-C is applied on layer 0 only, other layers start from zero, so the reduction yields alpha-AB plus beta-C exactly once.
`);
}

// ---------------------------------------------------------------- 7. Cost model & planner
{
  const s = pres.addSlide(); n++;
  title(s, 'Cost model and planner', 'Choose Pr, Pc, c and b at run time by minimising a calibrated α–β–γ model');
  s.addShape(pres.ShapeType.roundRect, { x: 0.5, y: 1.4, w: 9, h: 1.15, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  s.addText('T  =  α · [ (K/c)/b · (log₂Pr + log₂Pc) + log₂c ]   +   β·w · [ (K/c)·(M/Pr + N/Pc) + c·M·N/P ]   +   γ · 2MNK/P', { x: 0.7, y: 1.5, w: 8.6, h: 0.5, fontFace: 'Cambria', fontSize: 13.5, color: NAVY, align: 'center', isTextBox: true, margin: 0 });
  s.addText('α = message latency     β = inverse bandwidth     γ = time per flop (achieved)     w = word size     b = panel width', { x: 0.7, y: 2.05, w: 8.6, h: 0.4, fontFace: FONT, fontSize: 11, color: MUTED, align: 'center', isTextBox: true, margin: 0 });
  bullets(s, [
    { text: 'Planner enumerates every c | P, every Pr | (P/c), and b ∈ {32 … 1024}; drops configs over the per-rank memory budget; returns the minimiser', bold: true },
    { text: 'P ≤ a few thousand → the search costs microseconds', sub: true },
    { text: 'Closed-form sanity check: ignoring latency, c = 1 → Pr = √(P·M/N): the grid aspect ratio should match the output aspect ratio', bold: true },
    { text: 'Enumerator reproduces it: M = 65536, N = K = 1024, P = 64  →  64 × 1', sub: true },
    { text: 'α, β measured by ping-pong; γ by a panel-shaped run of the same kernel the run will use', bold: true },
    { text: 'Note: b appears only in the latency term — the model says b is purely a latency-vs-buffer knob. We test that later.', sub: true },
  ], { x: 0.5, y: 2.65, w: 9, h: 2.5, fontSize: 12.5 });
  footer(s, n);
  notes(s, `
[~1:30] The planner. The cost model is the standard alpha-beta-gamma form. Alpha is message latency, beta is inverse bandwidth, gamma is the achieved time per flop, w is the word size.

Three terms. Latency: number of broadcasts — K over c over b panels, each a log-depth broadcast along a row and a column — plus the final depth reduction. Bandwidth: the panel bytes each process receives, K over c times (M/Pr plus N/Pc), plus the 2.5D reduction of c copies of C. Compute: 2MNK over P.

The planner is an exhaustive enumeration. Every c that divides P, every Pr that divides P over c, every panel width from 32 to 1024. Discard anything over the memory budget, return the minimiser. P is at most a few thousand, so this costs microseconds.

A sanity check I insisted on: if you ignore latency and set c to 1, minimising K times (M/Pr + N/Pc) under Pr times Pc equals P gives Pr equals root of P·M/N. In words: the grid's aspect ratio should match the *output* matrix's aspect ratio. The enumerator reproduces that — for a tall 65536-by-1024 output on 64 processes it returns a 64-by-1 grid.

Calibration: alpha and beta from ping-pong, gamma from running the actual local kernel on a panel-shaped problem — the same kernel the real run uses. If you calibrate with a different kernel, the planner optimises for a machine that doesn't exist.

One thing to notice for later: b appears *only* in the latency term. The model says panel width is purely a latency-versus-buffer-memory knob. That's a prediction, and we'll see it's incomplete.
`);
}

// ---------------------------------------------------------------- 8. Kernel + verification
{
  const s = pres.addSlide(); n++;
  title(s, 'Local kernel and verification');
  s.addText('Three selectable kernels', { x: 0.5, y: 1.35, w: 4.4, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    { text: 'naive — textbook i-k-j loop', bold: true },
    { text: 'blocked — (MC, KC, NC) cache tiling, operands read in place', bold: true },
    { text: 'packed — A and B copied into contiguous 64-byte-aligned panels; 8×8 register tile kept in vector registers across the k loop', bold: true },
    { text: 'Panels zero-padded → micro-kernel always computes a full tile, writes back only the live corner: no edge branch in the hot loop', sub: true },
    { text: 'Threads own whole rows of C → no reduction, no false sharing', sub: true },
  ], { x: 0.5, y: 1.8, w: 4.4, h: 3.2, fontSize: 12.5 });
  s.addText('Three verification layers', { x: 5.2, y: 1.35, w: 4.3, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    { text: '1. Exact check vs. sequential reference from the same generator', bold: true },
    { text: '43-case suite: P ∈ {1…9} incl. primes, 61×53×47, 1×8 and 8×1 grids, P > M, α,β ≠ 1,0, b from 1 to 4096', sub: true },
    { text: '2. Norm-wise residual — never a floating-point equality test', bold: true },
    { text: '3. Distributed Freivalds test: random r, check A(Br) ≈ Cr in O(n²)', bold: true },
    { text: 'Cheap enough to leave on in production runs; vectors are O(n) and simply replicated', sub: true },
  ], { x: 5.2, y: 1.8, w: 4.3, h: 3.2, fontSize: 12.5 });
  footer(s, n);
  notes(s, `
[~1:15] Two supporting pieces: the local kernel and the verification.

Kernel. The distributed layer can't compensate for a bad inner loop, and a slow kernel inflates gamma until communication becomes invisible. So there are three selectable kernels. Naive: the textbook i-k-j loop. Blocked: cache tiling with operands read in place. Packed: copy A and B into contiguous, 64-byte-aligned panels and run an 8-by-8 register tile shaped so the compiler keeps the whole accumulator tile in vector registers across the k loop. Both packed panels are zero-padded, so the micro-kernel always computes a full tile and writes back only the live corner — no edge-case branch in the hot loop, no out-of-bounds access. Threading is over row blocks, so each thread owns whole rows of C: no reduction, no false sharing.

Verification has three layers. Layer 1: an exact comparison against a sequential reference built from the same deterministic generator. A 43-case suite covers P from 1 to 9 including primes, awkward sizes like 61 by 53 by 47, degenerate 1-by-8 and 8-by-1 grids, more processes than rows, non-trivial alpha and beta, and panel widths from 1 to 4096. Layer 2: a norm-wise residual — never floating-point equality. Layer 3: a distributed Freivalds test. Draw a random vector r, check that A times (B r) is approximately C r. That's O(n squared) work instead of O(n cubed), cheap enough to leave enabled in production runs where no reference exists.
`);
}

// ---------------------------------------------------------------- 9. Experimental setup
{
  const s = pres.addSlide(); n++;
  title(s, 'Experimental setup');
  const col = (x, h, items) => { s.addText(h, { x, y: 1.35, w: 2.9, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: NAVY, isTextBox: true, margin: 0 }); bullets(s, items, { x, y: 1.8, w: 2.9, h: 3.3, fontSize: 11.5 }); };
  col(0.5, 'Measurement platform', [
    'Single node: Intel Xeon 2.8 GHz, 4 physical cores, AVX-512, 15 GB RAM',
    'Ubuntu 24.04, GCC 13.3.0, Open MPI 4.1.6',
    'Ranks bound to cores, never oversubscribed',
    { text: 'Target: UniTN HPC cluster (~200 nodes, 15,284 cores, Omni-Path subset); 4 exclusive 72-core Xeon 6140M nodes per job, CPU type and network pinned', sub: true },
  ]);
  col(3.55, 'Build', [
    '-O3 -march=native -funroll-loops -ffp-contract=fast -fopenmp',
    'FMA contraction is disabled by GCC under -std=c11 — costs ~20%',
    { text: '-ffp-contract=fast is far weaker than -ffast-math (no reassociation) but changes the last bit; disclosed, not hidden', sub: true },
  ]);
  col(6.6, 'Protocol (Hoefler & Belli ’15)'.replace(' (Hoefler & Belli ’15)', ''), [
    '10 reps per config, each a separate launch with one untimed warm-up',
    'MPI_Barrier, then MPI_Wtime per rank, reduced with MPI_MAX',
    'Configs interleaved across the sweep, not run back-to-back',
    'One CSV row per rep; medians + bootstrap 95% CI computed afterwards; harmonic mean for rates; no outliers removed',
    { text: 'Each run sized to O(0.1 s) — shorter runs measure launch jitter', sub: true },
  ]);
  footer(s, n);
  notes(s, `
[~1:00] Setup, briefly, because the caveats matter for interpreting what follows.

Everything I'll show was measured on a single node: a 4-core Xeon at 2.8 GHz with AVX-512, Open MPI 4.1.6, ranks bound to cores, never oversubscribed. The code targets the University of Trento cluster — job scripts request four exclusive 72-core nodes with the CPU type and interconnect pinned so neither becomes an uncontrolled variable — but that multi-node campaign is queued, not complete. Keep that in mind: what follows validates the algorithms and the model, not the network.

Build flags: O3, march native, and ffp-contract=fast. I call that one out explicitly: GCC disables FMA fusion in strict C11 mode, which costs about 20%. Enabling it changes results in the last bit. It's far weaker than fast-math — no reassociation — but it's a semantic change so it's disclosed.

Protocol follows Hoefler and Belli's benchmarking guidelines. Ten repetitions per configuration, each a separate process launch with a warm-up. Barrier before the timer, time per rank reduced with MPI_MAX because the slowest rank defines the runtime. Configurations are interleaved so drift doesn't correlate with the variable under study. No averaging in C: one CSV row per repetition, medians and bootstrap confidence intervals computed afterwards, harmonic mean for rates, no outliers removed. Each run is sized to about a tenth of a second — shorter runs just measure launch jitter, which in an earlier iteration gave intervals a factor of several wide.
`);
}

// ---------------------------------------------------------------- 10. Kernel ablation
{
  const s = pres.addSlide(); n++;
  title(s, 'Result 1 — the kernel decides what we can measure');
  s.addImage({ path: plot('kernel_ablation.png'), x: 0.4, y: 1.35, w: 3.5, h: 2.35, sizing: { type: 'contain', w: 3.5, h: 2.35 } });
  s.addImage({ path: plot('roofline.png'), x: 3.9, y: 1.35, w: 3.2, h: 2.35, sizing: { type: 'contain', w: 3.2, h: 2.35 } });
  stat(s, 7.3, 1.4, 2.3, '30.6', 'Gflop/s, packed, 1 core (naive 7.55, tiled 9.83)');
  stat(s, 7.3, 2.8, 2.3, '4.05×', 'over naive · 46% of OpenBLAS · 34% of 2-FMA peak');
  bullets(s, [
    { text: 'M = N = 1024, K = 256 — one SUMMA panel. Flat thread scaling is a property of this 4-core node, not of the kernel.', bold: true },
    { text: 'Roofline: STREAM 11.9 GB/s → ridge at 5.6 flop/byte; this shape has 25.6. All kernels sit 4× past the ridge — the shortfall is vectorisation, not bandwidth. Gap to OpenBLAS is 2.2×, not an order of magnitude.', bold: true },
  ], { x: 0.5, y: 4.2, w: 9, h: 1.0, fontSize: 10.5 });
  footer(s, n);
  notes(s, `
[~1:00] First result: the kernel ablation, on a panel-shaped problem — 1024 by 1024 by 256, which is what one SUMMA step looks like.

Naive: 7.55 gigaflops per second. Cache-tiled: 9.83. Packed with the 8-by-8 register tile: 30.6. That's 4.05 times the naive kernel. As an external reference I measured single-threaded OpenBLAS dgemm on the same shape: 66.3 gigaflops — the dashed line — so the packed kernel is at 46% of a production BLAS. Since OpenBLAS exceeds the one-FMA-unit figure of 44.8, the core must issue two FMAs per cycle, giving a theoretical peak of 89.6, of which we reach 34%. Honest numbers for hand-written C.

The right-hand plot is a node roofline. Measured STREAM bandwidth is 11.9 gigabytes per second, which puts the ridge point at 5.6 flops per byte. This panel shape has an arithmetic intensity of 25.6 flops per byte against compulsory traffic — no hardware counters were available, so it's an algorithmic bound. All three kernels sit four times past the ridge, firmly compute-bound. So the distance from the naive kernel to the roof cannot be explained by memory traffic; it's instruction-level parallelism and vector utilisation in the inner loop. That also bounds what further tuning could buy: the remaining gap to OpenBLAS is 2.2 times, not the order of magnitude a memory-bound reading would suggest.

Why does this matter beyond the headline number? At 9.8 gigaflops the model's compute term dominates communication by two orders of magnitude. The planner would then be ranking configurations on differences of a fraction of a percent of predicted runtime. Fixing the kernel is a *precondition* for the decomposition to be measurable at all.

The absence of thread scaling in the plot — the orange bars — is because the node has only 4 cores and MPI ranks already use them; it's a property of this measurement box, not of the kernel.
`);
}

// ---------------------------------------------------------------- 11. Planner vs fixed grid
{
  const s = pres.addSlide(); n++;
  title(s, 'Result 2 — the planner earns its place on skewed shapes');
  s.addImage({ path: plot('planner_shapes.png'), x: 0.5, y: 1.35, w: 5.3, h: 3.5, sizing: { type: 'contain', w: 5.3, h: 3.5 } });
  const rows = [
    [{ text: 'Shape', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'Planner picks', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'vs fixed 2×2', options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ['short-fat (K = 32768)', '1×1×4  (pure k-parallel)', '1.36×  ✓'],
    ['wide B (M ≪ N)', '1×4×1', '1.24×  ✓'],
    ['tall A (M ≫ N)', '4×1×1', '1.12×  ✓'],
    ['tall-skinny', '2×2×1', '1.06×  (CIs overlap)'],
    ['square', '1×2×2', { text: '0.95×  (real loss)', options: { color: ACCENT, bold: true } }],
  ];
  s.addTable(rows, { x: 6.0, y: 1.4, w: 3.5, colW: [1.3, 1.15, 1.05], fontFace: FONT, fontSize: 9.5, color: INK, border: { type: 'solid', pt: 0.5, color: 'CCCCCC' }, rowH: 0.34 });
  s.addText('P = 4, medians of 10 runs with bootstrap 95% CIs. Three significant wins out of five — and the winning decompositions are ones a fixed 2D grid cannot express at all.', { x: 6.0, y: 4.05, w: 3.5, h: 1.0, fontFace: FONT, fontSize: 11, color: INK, isTextBox: true, margin: 0 });
  footer(s, n);
  notes(s, `
[~1:30] The main result: the planner against a fixed near-square grid, on five shapes at P equals 4. Blue is the fixed grid, orange is the planner.

(Note for you: the plot legend says "fixed 8x8 grid" — that's a label from the plotting script, but the experiment is at P=4 so the baseline is a fixed 2x2. If asked, say so.)

On short-fat operands — 512 by 512 output, K equals 32768 — the planner selects 1 by 1 by 4: pure k-parallelism, every process takes a quarter of the k dimension and the results are reduced at the end. That is 1.36 times faster, 0.228 seconds down to 0.167. A fixed 2D grid cannot reach this decomposition at all — it has no k axis.

For M much smaller than N it picks 1 by 4 by 1, 1.24 times faster. For M much larger than N, 4 by 1 by 1, 1.12 times. On those three the bootstrap confidence intervals are disjoint, so the gains are real. So it reproduces the analytic rule — grid aspect ratio equals output aspect ratio — from measured data rather than from the formula.

Tall-skinny gets 2 by 2 by 1 and is nominally 1.06 times faster, but the intervals overlap, so I claim no improvement there — only that the planner does no harm. Three significant wins out of five is the honest score.

And it *loses* on the square case, by 5%: it picks 1 by 2 by 2 instead of 2 by 2 by 1. The intervals are disjoint there too, so it's a real regression, not noise. I report that rather than hiding it, and the next slide explains why it happens — the cause is visible in the model itself.
`);
}

// ---------------------------------------------------------------- 12. Cost model accuracy + b sweep
{
  const s = pres.addSlide(); n++;
  title(s, 'Result 3 — the model, where it is right and where it is wrong');
  s.addImage({ path: plot('b_sweep.png'), x: 0.5, y: 1.35, w: 4.2, h: 3.15, sizing: { type: 'contain', w: 4.2, h: 3.15 } });
  const rows = [
    [{ text: 'b', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'total [s]', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'comm [s]', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'comp [s]', options: { bold: true, color: WHITE, fill: { color: NAVY } } }, { text: 'model / measured', options: { bold: true, color: WHITE, fill: { color: NAVY } } }],
    ['16', '0.2457', '0.0283', '0.2238', { text: '0.65 (under by 35%)', options: { color: ACCENT, bold: true } }],
    ['64', '0.1473', '0.0145', '0.1322', '1.08'],
    ['256', '0.1335', '0.0163', '0.1192', '1.20'],
    ['512', '0.1372', '0.0185', '0.1193', '1.16'],
  ];
  s.addTable(rows, { x: 4.9, y: 1.4, w: 4.6, colW: [0.5, 0.9, 0.9, 0.9, 1.4], fontFace: FONT, fontSize: 9.5, color: INK, border: { type: 'solid', pt: 0.5, color: 'CCCCCC' }, rowH: 0.3, align: 'center' });
  bullets(s, [
    { text: 'Calibrated α = 1.30 µs, 1/β = 4.16 GB/s, 1/γ = 28.7 Gflop/s. For b ≥ 64 the model is within 20% (threshold ±30%).', bold: true },
    { text: 'Model says only latency depends on b — but measured latency is negligible (21 µs). What varies is compute: 224 → 119 ms. Narrow panels = short, inefficient kernel calls.', bold: true },
    { text: 'Here b is a kernel-efficiency knob, not a latency knob. Same reason the planner is blind on the square case: 10 ms comm vs 150 ms compute — differences below its own accuracy.', bold: true },
  ], { x: 4.9, y: 3.35, w: 4.6, h: 1.8, fontSize: 10 });
  s.addText('P = 4, n = 2048', { x: 0.5, y: 4.55, w: 4.2, h: 0.3, fontFace: FONT, fontSize: 10, color: MUTED, align: 'center', isTextBox: true, margin: 0 });
  footer(s, n);
  notes(s, `
[~1:30] How good is the cost model? Calibration on this node gives alpha 1.3 microseconds, bandwidth 4.16 gigabytes per second, and 28.7 gigaflops per second for gamma.

Evaluating the model at P equals 4, n equals 2048 against measured medians: for panel widths 64, 128, 256 and 512 the model-over-measured ratios are 1.08, 1.15, 1.20, 1.16. Within 20%, comfortably inside the plus-or-minus 30% acceptance threshold I set beforehand.

At b equals 16, though, the model *under*-predicts by 35%. The table shows why. The model says only the latency term depends on b. The measured communication time is indeed small and flat — around 15 to 28 milliseconds, with the pure latency part just 21 microseconds. What actually varies with b is *compute*: 224 milliseconds at b equals 16, down to 119 at b equals 256. Narrow panels mean short, inefficient kernel invocations — the packed kernel can't amortise its packing cost on a 16-wide panel.

So the textbook reading of b as a pure latency knob is wrong on this machine. b is dominated by local kernel efficiency, and a model that omits that term mispredicts by a third. That's a concrete limitation of the model that the measurements expose.

The same thing explains the 5% regression on the square case from the previous slide: there, communication is 10 milliseconds against 150 of compute, so the planner is discriminating between candidates that differ by under 1% of predicted runtime — below the model's own accuracy. A refinement would be to fall back to the conventional grid whenever the predicted spread across candidates is smaller than the model's residual error.
`);
}

// ---------------------------------------------------------------- 13. Scaling, engines, c sweep
{
  const s = pres.addSlide(); n++;
  title(s, 'Result 4 — scaling, engines, replication (single node)'.replace(' (single node)', ''));
  s.addImage({ path: plot('strong_scaling.png'), x: 0.4, y: 1.3, w: 4.6, h: 2.0, sizing: { type: 'contain', w: 4.6, h: 2.0 } });
  s.addImage({ path: plot('engine_comparison.png'), x: 0.4, y: 3.3, w: 4.6, h: 2.0, sizing: { type: 'contain', w: 4.6, h: 2.0 } });
  s.addImage({ path: plot('c_sweep.png'), x: 5.1, y: 1.3, w: 4.4, h: 1.9, sizing: { type: 'contain', w: 4.4, h: 1.9 } });
  bullets(s, [
    { text: 'Strong scaling, n = 2048: speedup 1.00 / 1.92 / 3.55 at P = 1, 2, 4 → 89% efficiency, 124.5 Gflop/s. Shared-memory result: says nothing about the network.', bold: true },
    { text: 'Engines, P = 4: SUMMA 120.1, 2.5D 117.3, naive1D 95.4 Gflop/s. naive1D moves 1.5× the bytes (K·N per rank, independent of P). Cannon 124.3 — only on the square divisible case.', bold: true },
    { text: 'c sweep: 0.138 / 0.145 / 0.225 s for c = 1, 2, 4. P^(1/3) = 1.59 lies between the feasible c = 1 and 2 — a decisive test needs larger P.', bold: true },
  ], { x: 5.1, y: 3.3, w: 4.4, h: 1.85, fontSize: 10 });
  footer(s, n);
  notes(s, `
[~1:15] Three smaller results, all single-node.

Strong scaling at n equals 2048: speedups of 1.00, 1.92 and 3.55 at 1, 2 and 4 processes — 89% parallel efficiency at four, 124.5 gigaflops per second. I stress this is a shared-memory result: the "network" here is memory bandwidth. It validates the algorithms and the model but says nothing about real network behaviour.

Engine comparison at P equals 4: SUMMA reaches 120, 2.5D 117, naive-1D 95 gigaflops per second. The right-hand panel shows why: naive 1D moves 0.101 gigabytes per rank against SUMMA's 0.067 — 1.5 times the traffic, because its per-rank volume is K times N regardless of P. It doesn't scale. Cannon is competitive — 124 — on the square, divisible instance it is restricted to: marginally faster where it applies, inapplicable everywhere else. That's the measured argument for SUMMA as the base engine.

The replication sweep: c equals 1, 2, 4 gives 0.138, 0.145 and 0.225 seconds. With P equals 4, the theoretical optimum P to the one-third is 1.59, which lies between the feasible values 1 and 2 — so on this node replication can't help, and a decisive test of the c equals P-to-the-one-third rule needs a larger P on the cluster.
`);
}

// ---------------------------------------------------------------- 14. SHM broadcast + numerics
{
  const s = pres.addSlide(); n++;
  title(s, 'Two more findings: a shared-memory broadcast, numerics');
  s.addText('Node-aware two-level broadcast (MPI-3 SHM)', { x: 0.5, y: 1.35, w: 5.4, h: 0.4, fontFace: FONT, fontSize: 14, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    { text: 'Alternative to MPI-4 persistent collectives: one leader per node joins the inter-node broadcast; the other ranks read the panel from shared memory', bold: true },
    { text: 'Modelled off-node panel traffic halves with 2 ranks/node: 0.0168 → 0.0084 GB, as predicted', bold: true },
    { text: 'But the shared panel becomes a critical section: a fast rank can overwrite a panel a slower neighbour is still reading → barrier before each reuse', bold: true },
    { text: 'Those barriers are exactly what pipelining avoids → does not compose with lookahead', sub: true },
    { text: 'Timing comparison is meaningless on one node — byte reduction reported, speed-up deliberately not claimed', sub: true },
  ], { x: 0.5, y: 1.8, w: 5.4, h: 3.3, fontSize: 12 });
  s.addShape(pres.ShapeType.roundRect, { x: 6.2, y: 1.35, w: 3.3, h: 3.6, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.08 });
  s.addText('Numerical behaviour', { x: 6.4, y: 1.5, w: 3.0, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  s.addText('8.7e-16 – 1.1e-15', { x: 6.4, y: 2.0, w: 3.0, h: 0.6, fontFace: FONT, fontSize: 22, bold: true, color: ACCENT, isTextBox: true, margin: 0 });
  s.addText('relative residual across {engine, c, b}', { x: 6.4, y: 2.6, w: 3.0, h: 0.4, fontFace: FONT, fontSize: 11, color: MUTED, isTextBox: true, margin: 0 });
  s.addText('Summation order changes with c and b, so results are not bitwise reproducible across configurations. Quantified rather than hidden: consistent with O(Kε) error growth.', { x: 6.4, y: 3.1, w: 3.0, h: 1.7, fontFace: FONT, fontSize: 11.5, color: INK, valign: 'top', isTextBox: true, margin: 0 });
  footer(s, n);
  notes(s, `
[~1:00] Two shorter findings.

First, a broadcast mechanism. As an alternative to MPI-4 persistent collectives, I implemented a node-aware two-level broadcast on an MPI-3 shared-memory window: one leader per node participates in the inter-node broadcast, and the node's other ranks read the panel directly out of shared memory. With two ranks per node in the relevant communicator, modelled off-node traffic halves — 16.8 to 8.4 megabytes — exactly as predicted.

But there's a catch worth reporting. Making the panel shared turns a private buffer into a critical section: a fast rank can overwrite a panel that a slower neighbour is still reading. So you need a barrier before each reuse — and those barriers are exactly what pipelining with lookahead tries to avoid. The technique doesn't compose with lookahead. And a timing comparison is meaningless on a single node, so I report the mechanism and the byte reduction and deliberately do not claim a speed-up.

Second, numerics. Summation order changes with c and b, so results aren't bitwise reproducible across configurations. Rather than hide that, I quantify it: relative residuals across all engines, replication depths and panel widths span 8.7e-16 to 1.1e-15, consistent with error growing like K times machine epsilon. That's the expected behaviour for double precision.
`);
}

// ---------------------------------------------------------------- 15. Conclusions
{
  const s = pres.addSlide(); n++;
  title(s, 'Conclusions and honest limitations');
  s.addText('What was shown', { x: 0.5, y: 1.35, w: 4.4, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  bullets(s, [
    { text: 'A distributed GEMM can be genuinely generic — any M, N, K; any P incl. primes; any grid — without giving up performance', bold: true },
    { text: 'The process grid is worth choosing at run time: a few-dozen-line planner gives a significant 1.12×–1.36× over a fixed grid on three of five shapes', bold: true },
    { text: 'It finds decompositions (1×1×4) a fixed 2D grid cannot express', sub: true },
    { text: 'Cost model within 20% of measured runtime for b ≥ 64', bold: true },
    { text: '43/43 tests, Freivalds at scale, residuals ~1e-15', bold: true },
  ], { x: 0.5, y: 1.8, w: 4.4, h: 3.3, fontSize: 12.5 });
  s.addText('Three limitations', { x: 5.2, y: 1.35, w: 4.3, h: 0.4, fontFace: FONT, fontSize: 16, bold: true, color: ACCENT, isTextBox: true, margin: 0 });
  bullets(s, [
    { text: '1. Planner regresses 5% on square operands — it ranks differences below its own accuracy. Fix: gate on predicted spread vs. residual error', bold: true },
    { text: '2. Model attributes all b-dependence to latency; the dominant effect is kernel efficiency. Fix: a panel-width term in γ', bold: true },
    { text: '3. Results are single-node. They validate the algorithms, the model and the correctness argument; the multi-node campaign is queued, not complete. β across a real interconnect will differ by ~an order of magnitude', bold: true },
  ], { x: 5.2, y: 1.8, w: 4.3, h: 3.3, fontSize: 12.5 });
  footer(s, n);
  notes(s, `
[~1:15] To conclude.

What was shown: a distributed GEMM can be genuinely generic — arbitrary M, N, K, arbitrary P including primes, arbitrary grids — without giving up performance. And the process grid is worth choosing at run time rather than fixing by convention: a cost-model planner that costs a few dozen lines gives a statistically significant 1.12 to 1.36 times over a fixed near-square grid on three of the five shapes, and finds decompositions like 1-by-1-by-4 that a fixed 2D grid can't even express. The model predicts runtime within 20% where the panel width isn't pathologically small. Correctness is backed by 43 exact tests, a Freivalds check that runs at full size, and residuals around 1e-15.

The honest limitations are three. First, the planner regresses by 5% on square operands, where the differences it ranks are below its own accuracy; gating it on the predicted spread would fix this. Second, the model attributes all b-dependence to latency, whereas the measured dominant effect is local kernel efficiency — adding a panel-width term to gamma is the obvious refinement. Third, and most important: the results are single-node. They validate the algorithms, the model and the correctness argument, but the multi-node campaign that would test the communication claims at scale is queued rather than complete, and beta measured across a real interconnect will differ from the intra-node value by roughly an order of magnitude.
`);
}

// ---------------------------------------------------------------- 16. Future work + reproducibility
{
  const s = pres.addSlide(); n++;
  s.background = { color: NAVY };
  s.addText('Future work and reproducibility', { x: 0.5, y: 0.35, w: 9, h: 0.7, fontFace: FONT, fontSize: 28, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  s.addText('Next steps', { x: 0.5, y: 1.3, w: 4.4, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: 'CADCFC', isTextBox: true, margin: 0 });
  s.addText([
    'Run the multi-node campaign (jobs/*.pbs, 4 × 72-core nodes)',
    'Type genericity (G3): float/double via include-template — halves volume in the bandwidth-bound regime',
    'MPI-4 persistent collectives as a third broadcast policy',
    'Network roofline: I_net = (2MNK/P) / bytes',
    'Compare against COSMA on the same shapes — the correct reference for the planner claim',
  ].map((t, i, a) => ({ text: t, options: { bullet: true, breakLine: i < a.length - 1, paraSpaceAfter: 6 } })), { x: 0.5, y: 1.75, w: 4.4, h: 3.0, fontFace: FONT, fontSize: 12.5, color: WHITE, valign: 'top', isTextBox: true, margin: 0 });
  s.addText('Reproduce it', { x: 5.2, y: 1.3, w: 4.3, h: 0.4, fontFace: FONT, fontSize: 15, bold: true, color: 'CADCFC', isTextBox: true, margin: 0 });
  s.addText([
    'github.com/elliott-dp/parallel-programming-project  (b95b92c, tag v1.0)',
    'make && ./tests/run_tests.sh   # 43/43',
    './scripts/run_local.sh         # single-node runs',
    'scripts/plot_results.py <csv> --kind <k>',
    'make roofline BLAS=1           # measure roofs',
    'qsub jobs/smoke.pbs            # cluster build+verify',
    '',
    'Every figure regenerates from committed CSVs; bootstrap is seeded → byte-identical. docs/HPC.md, docs/EXPERIMENTS.md map each experiment to its figure.',
  ].map((t, i, a) => ({ text: t, options: { breakLine: i < a.length - 1, fontFace: i < 6 ? 'Courier New' : FONT, fontSize: i < 6 ? 9.5 : 11 } })), { x: 5.2, y: 1.75, w: 4.3, h: 3.0, color: WHITE, valign: 'top', isTextBox: true, margin: 0 });
  s.addText('Thank you — questions?', { x: 0.5, y: 4.85, w: 9, h: 0.5, fontFace: FONT, fontSize: 18, bold: true, color: WHITE, isTextBox: true, margin: 0 });
  notes(s, `
[~0:45] Future work, in priority order. First, actually run the multi-node campaign — the job scripts exist. Second, type genericity: float and double through one code path via an include-template; float halves communication volume in the bandwidth-bound regime. Third, MPI-4 persistent collectives as a third broadcast policy. Fourth, a network roofline that places each configuration by its arithmetic intensity against the network. Fifth, a comparison against COSMA on the same shapes, which is the correct reference point for the planner claim.

Everything is reproducible: source, job scripts, all raw CSVs and every figure are in the repository. make and run_tests gives 43 of 43. run_local reproduces the single-node campaign. Every figure regenerates from the committed CSVs with one command, and the bootstrap is seeded, so regeneration is byte-identical.

Thank you — happy to take questions.

--- LIKELY QUESTIONS ---
Q: Why not just use ScaLAPACK / a library? — The point was to study decomposition choice; a library hides it. Also block-cyclic is unnecessary for uniform GEMM work.
Q: Why is the packed kernel only ~half of OpenBLAS? — Hand-written C micro-kernel relies on the compiler for vectorisation; no assembly, no prefetch tuning. 68% of single-core FMA peak is the honest figure.
Q: Why does the planner lose on square? — The candidates differ by <1% of predicted time; below model accuracy. Fix: fall back to near-square grid when predicted spread < residual error.
Q: Is 2.5D ever helpful here? — Not on 4 processes: P^(1/3)=1.59, no feasible c between 1 and 2. Needs larger P.
Q: How do you verify at full size? — Freivalds: random r, check A(Br) ≈ Cr, O(n^2). Probability of a false pass is ≤ 1/2 per trial, trials are independent.
Q: How is a prime P handled? — Grid becomes 1×P or P×1 (or with c). Uneven blocks handle any dimension. Test suite includes P=2,3,5,7.
Q: What does -ffp-contract=fast change? — Fuses a*b+c into one FMA (one fewer rounding). No reassociation. Changes last bit; ~20% faster.
`);
}

const out = path.join(__dirname, 'presentation.pptx');
pres.writeFile({ fileName: out }).then(() => console.log('wrote', out));
