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
[0:30]
Good morning. My project is called gemm2d.

It's a matrix multiplication that runs across many machines at once.

Two things make it different. It works for any matrix shape and any number of machines. And it decides how to split the work by itself, at run time.

I'll show you the problem, then the design, then five results, and then what's still missing.
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
[1:00]
Matrix multiply is the operation almost everything reduces to. Linear algebra, and most of deep learning.

On one machine it's about arithmetic. Across many machines it's about moving data. And how much data moves depends entirely on how you split the matrices up.

Now, the word generic in my title is the whole problem.

A textbook implementation makes three assumptions. And all three are wrong in practice.

First, it assumes square matrices. But real problems are lopsided. Take a Gram matrix, A-transpose times A. Ten thousand samples, a hundred features. The output is only a hundred by a hundred, but the inner dimension is ten thousand. Tiny result, enormous amount of work.

Second, it assumes the number of processes is a perfect square. Four, nine, sixteen. But the scheduler gives you whatever is free. Sometimes seven. And seven has no square grid.

Third, it assumes the sizes divide evenly. They don't. The usual fix is to pad with zeros, and that's worse than it looks. You waste work, but more importantly you hide bugs. Because zeros contribute nothing, so a wrong answer still looks plausible.

So the question is: can we drop all three, and not lose performance?

--- IF ASKED ---
Why not just pad? It costs flops, but the real reason is debuggability: padded regions contribute zero, so an indexing bug gives a plausible wrong answer instead of an obviously wrong one.
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
[0:45]
Let me be precise about what generic means. I split it into three.

G1 is shape. Any M, N, K. Done and tested.

G2 is the machine. Any number of processes, including primes. Any grid. Sizes that don't divide. Also done and tested.

G3 is the data type, the same code working for float and double. I didn't do that one. It's double only.

Then four goals. Correctness for any input. A local kernel fast enough that my measurements actually show the communication. A planner that chooses the layout from measurements instead of convention. And a way to check correctness even on problems too big to verify directly.

And things I explicitly did not do: sparse matrices, Strassen-style algorithms, GPUs, fault tolerance.

--- WATCH OUT ---
Alpha and beta mean two different things in this project. Here they are the scalars in C = alpha AB + beta C. On slide 7 they are latency and inverse bandwidth. Know which slide you are on.

--- IF ASKED ---
Why is the kernel work relevant to a parallel computing project? If the kernel is slow, compute dominates communication by a hundred times, so every timing would just measure the inner loop. A fast kernel is a precondition for communication to be visible at all.
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
[1:15]
A quick tour of what already exists. Each of these justifies one of my decisions.

Cannon's algorithm, from 1969. It arranges the processes in a square and rotates the data step by step. Only neighbour-to-neighbour messages, very efficient. But it only works on a square grid with sizes that divide. Which is exactly what I refuse to assume. So it can't be my base.

SUMMA, from 1997. Instead of rotating, it broadcasts thin slices along rows and columns. The slice width is independent of the grid, and that's what makes it flexible. So that's my base engine.

ScaLAPACK deals its blocks out round-robin, like cards. That exists to balance the work in factorisations, where the work shrinks as you go. Matrix multiply has uniform work, so I don't need it. I use plain blocks and avoid the complicated indexing.

Then there's theory saying: if memory is tight, 2D is already optimal. But if you have spare memory, you can trade it for less communication. That's what 2.5D does. It splits the K dimension across layers.

And finally COSMA, from 2019. That's the state of the art. It finds a near-optimal layout for any shape and any machine.

So where is my contribution? COSMA is complicated. And meanwhile, most real code still hard-codes a square grid. So my question is: how much of that benefit do you get from something small enough that people would actually use it? A few dozen lines that try every possible grid and pick the cheapest.

It's a selector. I am not claiming it's optimal.

--- IF ASKED ---
Why not just use COSMA? For production you should. My question was what a minimal component recovers, because that is what gets adopted. A head-to-head against COSMA is first in my future work.
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
[1:15]
Now the design. All four of my algorithms share one data layout.

Each process holds a descriptor: how big the whole matrix is, how big its own piece is, and where its piece starts.

When the sizes don't divide, I don't pad. I just give some processes one extra row. Process zero gets three hundred and thirty-four rows, the others get three hundred and thirty-three.

That sounds minor, but it's the most useful decision in the whole codebase. Every awkward case, a ragged last block, a one-by-P grid, more processes than rows, all of them come down to that single function. So I test that one function hard, and everything else follows.

Second thing: nothing is created on process zero and then distributed. Each process generates its own piece from a hash of the coordinates. So element i,j has the same value no matter how many processes you run. That removes a bottleneck, and it means I can compare results across configurations exactly.

And on the right, the four engines. Naive 1D and Cannon are baselines. Note that Cannon refuses to run on cases it can't handle, rather than padding, and that refusal is my evidence for choosing SUMMA. SUMMA is the main one. And 2.5D adds the third dimension.
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
[1:15]
This is the SUMMA loop.

The idea: walk along the K dimension in thin slices. At each step, whoever owns that slice of A broadcasts it along its row of processes. Whoever owns that slice of B broadcasts it down its column. Then everyone multiplies the two thin slices and adds the result into their block. Move to the next slice, repeat.

Now here's the one subtle thing that makes it generic.

A's K dimension is divided among the columns of the grid. B's K dimension is divided among the rows. Those are two different divisions, unless everything is square and divides evenly.

So a slice can cross a boundary in A, or in B, or both. And that's line five: I clip every step to the overlap of both owners' ranges. That's the whole trick. It's what lets any grid and any size work.

On the right, the 2.5D engine. Each layer takes a chunk of K, runs a complete SUMMA on it, and produces a partial result. Then one reduction adds the partial results together.

I chose this form because it reuses the SUMMA code unchanged, about forty extra lines. And the only extra memory is one copy of C per layer.
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
[1:30]
This is the model the planner minimises. Three terms.

The first is latency. How many messages you send, times the cost of one message.

The second is bandwidth. How many bytes you send, divided by the transfer speed.

The third is computation. How many operations, divided by how fast you do them.

The three constants, alpha beta gamma, I measure on the actual machine before running. Latency and bandwidth with a ping-pong test. And the compute speed by running the real kernel, the same one the run will use. If you calibrate with a different kernel, you're optimising for a machine that doesn't exist.

Then the planner just tries everything. Every way of factoring the process count into a grid, every panel width. Throws away anything that doesn't fit in memory. Returns the cheapest. And that costs microseconds.

I also checked it against theory. If you ignore latency, the maths says the grid shape should match the output shape. And the planner reproduces that on its own, from the numbers.

One thing to notice for later: the panel width only appears in the latency term. So the model claims it's purely a latency knob. Hold on to that. It turns out to be wrong.
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
[1:15]
Two supporting pieces.

First the local kernel. I have three versions. A textbook triple loop. A cache-blocked one. And a packed one, which copies the data into contiguous aligned buffers and uses an eight-by-eight block that stays in the CPU registers for the whole inner loop.

Small detail I'm proud of: the buffers are padded with zeros, so the inner loop always computes a full block and only writes back the part that's real. That means no branches in the hot loop, and no risk of reading out of bounds.

And threads take whole rows, so no two threads ever write the same output. No locks, no false sharing.

Second, correctness. Three layers.

Layer one: compare against a sequential reference. Forty-three test cases. Prime process counts, ugly sizes like sixty-one by fifty-three by forty-seven, degenerate grids, more processes than rows, panel widths from one to four thousand.

Layer two: I compare norms, never exact equality. Floating point never gives exact equality.

Layer three, and this is the interesting one: Freivalds' test. Pick a random vector r. Check that A times B times r equals C times r. That's n-squared work instead of n-cubed. So it's cheap enough to leave switched on in real runs, where there is no reference answer to compare against.
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
[1:00]
About the measurements.

Everything I'm about to show was run on a single node. Four cores, one Xeon. Processes pinned to cores, never oversubscribed.

The code targets the Trento cluster, and the job scripts are written, but those runs are still queued. So keep that in mind. What follows validates the algorithms and the model. It says nothing about a real network.

On the build, one flag I want to declare. I enable FMA contraction. It's about twenty percent faster, and it changes the very last bit of the result. It's much weaker than fast-math, no reordering of arithmetic. But it is a semantic change, so I say so rather than hiding it.

And the measurement protocol follows the standard guidelines. Ten repetitions, each a fresh process launch, with a warm-up first. A barrier before the timer. And I take the slowest process, because the slowest one is what you actually wait for.

I never average inside the C code. Every repetition goes to a CSV, and the statistics happen afterwards. Medians and bootstrap confidence intervals. No outliers removed.

And every run is sized to take about a tenth of a second. Anything shorter and you're just measuring process startup.
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
[1:00]
Let's start with the local kernel. That's the code doing the actual multiply inside one process.

I wrote three versions. The naive one gets seven and a half gigaflops. The cache-blocked one, about ten. And the packed one, thirty.

So four times faster than naive.

For reference, I measured OpenBLAS on the same problem. It gets sixty-six. So I'm at about half of a professional library. For hand-written C, I'll take that.

Now, why am I showing you a single-core result in a parallel computing talk?

Because if the kernel is slow, computation swamps everything else. At ten gigaflops, compute is a hundred times bigger than communication. Every graph after this one would just be measuring my inner loop.

So a fast kernel isn't a bonus here. It's what makes the rest measurable.

On the right is a roofline. It tells you whether you're limited by memory or by the processor. The turning point is at five; my problem sits at twenty-five. All three kernels are far to the right of it. So they're compute-bound. The gap to the roof is about vectorisation, not memory.

One last note: the flat bars for four threads are just this machine. It only has four cores, and the MPI processes are already using them.
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
[1:30]  ** HEADLINE SLIDE. SLOW DOWN. **
This is the main result.

I ran five different matrix shapes on four processes. Blue is the standard fixed two-by-two grid. Orange is my planner choosing for itself.

Start with the first one, short and fat. Small output, huge inner dimension.

The planner picks one-by-one-by-four. Meaning: don't split the output at all. Split K four ways instead.

That's thirty-six percent faster. And notice, a fixed 2D grid can't even express that choice. It has no K axis.

Two more. When the matrix is wide, it picks a one-by-four grid, twenty-four percent faster. When it's tall, four-by-one, twelve percent.

For those three the confidence intervals don't overlap. So those wins are real.

The fourth one, tall-skinny, is nominally six percent faster, but the intervals overlap. So I don't claim it. I only claim it does no harm.

And the fifth one, square matrices, it actually loses. Five percent slower. And that one's real too.

So: three real wins out of five. Three is the honest score, not five.

The next slide explains the loss.

--- NOTE ---
The chart legend says fixed 8x8 grid. That is a wrong label from the plotting script. The experiment is at P=4, so the baseline is 2x2. Say so if anyone spots it.
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
[1:30]
So how good is the model?

First I measure the machine. Latency, one microsecond. Bandwidth, four gigabytes a second. Compute, about thirty gigaflops.

Then I compare prediction against reality. For panel widths from sixty-four up to five hundred, the model is within twenty percent. I'd fixed thirty percent as my threshold beforehand. So it passes.

But at panel width sixteen it's off by thirty-five percent. And the reason is interesting.

My model says the panel width only affects latency.

Now look at the table. The communication column barely moves. And the pure latency part is twenty microseconds. Nothing at all.

What actually changes is the compute column. Two hundred and twenty milliseconds, down to one hundred and twenty.

Because a narrow panel means lots of tiny kernel calls. And my kernel needs a wide panel to be efficient.

So the textbook says panel width is a latency knob. On this machine, it isn't. It's a kernel-efficiency knob. And leaving that out costs me a third.

And that's also why the planner lost on square matrices. There, communication is ten milliseconds against a hundred and fifty of compute. So it's ranking options that differ by less than one percent, which is below its own accuracy.

It's essentially guessing between near-ties.
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
[1:15]
Three quicker results. All on one node.

First, scaling. One, two, four processes gives speedups of one point nine and three point five. That's eighty-nine percent efficiency at four.

But let me be clear, this is a single node. The network here is really just memory. It says nothing about a real cluster.

Second, the engines. SUMMA gets a hundred and twenty gigaflops. 2.5D, a hundred and seventeen. Naive 1D, only ninety-five.

The chart on the right shows why. Naive 1D moves one and a half times more data. And that's the point from the beginning: its traffic doesn't shrink when you add processes.

Cannon is actually the fastest, at a hundred and twenty-four. But only on the one case it's allowed to run, square and divisible. Slightly faster where it works, useless everywhere else.

That's exactly why I built on SUMMA.

Third, the replication depth. One is best, two slightly worse, four much worse.

That's expected. With only four processes, theory puts the optimum around one point six, so there's no useful value to pick. Testing that properly needs a bigger machine.
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
[1:00]
Two smaller findings.

The first is a broadcast optimisation. Normally every process receives the panel over the network. Instead, one process per node receives it, and the others read it straight out of shared memory.

That halves the data crossing the network. And I verified that number.

But there's a catch worth telling you. Once the panel is shared, it's a shared resource. A fast process can overwrite a panel that a slower neighbour is still reading. So you need a barrier before reusing it.

And barriers are exactly what overlapping communication is supposed to avoid. So this trick doesn't combine with pipelining.

And since I only have one node, a timing comparison would be meaningless. So I report the mechanism and the byte saving, and I don't claim a speedup.

The second finding is numerical. When you change the configuration, the additions happen in a different order. So results aren't bit-identical across configurations.

Rather than hide that, I measured it. The error is around ten to the minus fifteen everywhere. That's exactly what you expect for double precision at this size.
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
[1:15]
So, to conclude.

A distributed matrix multiply can be genuinely generic. Any shape, any number of processes, any grid. Without giving up performance.

And the layout is worth choosing at run time. My planner is a few dozen lines, and it wins between twelve and thirty-six percent on three of the five shapes I tested. And it finds layouts a fixed 2D grid simply cannot express.

The model predicts real runtime within twenty percent, as long as the panel isn't tiny.

And correctness is solid. Forty-three tests pass, and the Freivalds check works at full scale.

Now the three limitations, and I want to be straight about them.

One. The planner loses five percent on square matrices, because there it's ranking options that differ by less than its own accuracy. The fix is simple: if the predictions are too close together, fall back to the conventional grid.

Two. My model blames the panel width on latency, when the real effect is kernel efficiency. The fix is to make the compute term depend on panel width.

Three, and this is the big one. Everything here is single node. It validates the algorithms, the model and the correctness. But it does not test the communication claims at scale. The multi-node runs are queued, not finished. And on a real network the bandwidth constant will be about ten times worse than what I measured here.
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
[0:45]
Future work, in the order I'd do it.

First: actually run the cluster campaign. The scripts are written.

Second: support float as well as double. That halves the data volume, which matters exactly when you're bandwidth-limited.

Third: persistent collectives from MPI-4, as a third broadcast option.

Fourth: a network roofline, to see which configurations are network-limited.

And fifth: compare against COSMA on the same shapes. That's the proper reference point for my claim, and it's the experiment I'd most want to run next.

Everything is reproducible. The source, the job scripts, the raw CSVs and every figure are in the repository. The tests build and pass with one command. And every figure regenerates from the committed data with a fixed random seed, so you get exactly the same picture.

Thank you. I'm happy to take questions.

=========================== Q&A CHEAT SHEET ===========================

Why not use ScaLAPACK or a library?
  The point was to study the decomposition choice, which a library hides. And block-cyclic
  exists for LU/QR load balance; GEMM work is uniform, so plain blocks are simpler and faster
  to index.

Your kernel is only half of OpenBLAS. Isn't that the bottleneck?
  It's hand-written C, no assembly. The roofline shows it's compute-bound, four times past
  the ridge, so the gap is vectorisation, not memory. What matters here is that it's fast
  enough for communication to be visible. Swapping in BLAS would only change gamma.

Why does the planner lose on square matrices? Doesn't that break your claim?
  On square shapes all candidates are within one percent in the model, below the model's own
  accuracy, so it's picking among near-ties. Fix: if the predicted spread is smaller than the
  residual error, keep the conventional grid. The wins are on skewed shapes, where candidates
  differ by tens of percent.

How does a prime P work?
  The only factorisations are 1xP or Px1, times c. Uneven blocks handle any dimension.
  Tested at P = 2, 3, 5, 7.

Is 2.5D ever useful in your results?
  Not at four processes: P to the one-third is 1.59, so no feasible c between 1 and 2. But
  it's what makes the 1x1x4 win possible, that IS the 2.5D engine. A real c sweep needs 64+.

How do you verify a result too big for a reference?
  Freivalds. Random vector r, check A(Br) is approximately Cr, n-squared work. A wrong C
  passes one trial with probability at most one half; trials are independent.

What does ffp-contract=fast change? Is it cheating?
  It fuses a*b+c into one FMA, one rounding instead of two. No reassociation, so it is not
  fast-math. Changes the last bit, gains about twenty percent, and can be turned off with
  one flag.

Why single node only?
  The multi-node campaign is scripted, six PBS jobs, but hadn't run in time. Single-node
  results validate correctness, algorithms and calibration. They cannot validate the
  communication claims, and I'd rather say that than present memory bandwidth as if it were
  an interconnect.

Why the k-slab 2.5D instead of the original Solomonik-Demmel?
  The original replicates A and B across layers and needs its own schedule. The k-slab form
  runs the existing SUMMA on a slice of K per layer plus one reduce. Forty lines, same
  asymptotics for GEMM, extra memory is only c copies of C.

What about float support?
  Not implemented, scalar_t is a compile-time double. Plan is an include-template plus an
  MPI type trait. Float halves communication volume, which is why it's first in future work
  after the cluster runs.

Is the shared-memory byte saving measured?
  No, it's a model: panel bytes the policy asks the network for. No hardware counters were
  available. I say so and don't claim a timing win.

Isn't your planner just a worse COSMA?
  Yes, deliberately. Smaller search space, no optimality proof, but calibrated to the actual
  machine and about forty lines. The contribution is the cost-benefit ratio, not beating COSMA.
`);
}

const out = path.join(__dirname, 'presentation.pptx');
pres.writeFile({ fileName: out }).then(() => console.log('wrote', out));
