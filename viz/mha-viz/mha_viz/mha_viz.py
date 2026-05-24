"""Interactive Multi-Head Attention Visualizer with Conceptual Explanations."""

import math
import random
import reflex as rx


def softmax(vals: list[float]) -> list[float]:
    if not vals:
        return []
    max_val = max(vals)
    exps = [math.exp(v - max_val) for v in vals]
    total = sum(exps)
    return [e / total for e in exps]


def make_matrix(rows: int, cols: int, lo: float = -2.0, hi: float = 2.0) -> list[list[float]]:
    return [[random.uniform(lo, hi) for _ in range(cols)] for _ in range(rows)]


def matmul_2d(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows_a, cols_a = len(a), len(a[0])
    cols_b = len(b[0])
    return [
        [sum(a[i][k] * b[k][j] for k in range(cols_a)) for j in range(cols_b)]
        for i in range(rows_a)
    ]


def transpose(m: list[list[float]]) -> list[list[float]]:
    if not m:
        return []
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class MHAState(rx.State):
    # --- Parameters ---
    seq_len: int = 4
    embed_dim: int = 6
    num_heads: int = 2
    step: int = 0

    # --- Data ---
    query_input: list[list[float]] = []
    key_input: list[list[float]] = []
    value_input: list[list[float]] = []

    wq: list[list[float]] = []
    wk: list[list[float]] = []
    wv: list[list[float]] = []
    wo: list[list[float]] = []

    q_proj: list[list[float]] = []
    k_proj: list[list[float]] = []
    v_proj: list[list[float]] = []

    q_heads: list[list[list[float]]] = []
    k_heads: list[list[list[float]]] = []
    v_heads: list[list[list[float]]] = []

    attn_weights: list[list[list[float]]] = []
    attn_output_heads: list[list[list[float]]] = []

    merged: list[list[float]] = []
    output: list[list[float]] = []

    # --- Explanation data ---
    analogy: str = ""
    why: str = ""
    what_if: str = ""
    key_insight: str = ""

    def _set_explanation(self):
        explanations = {
            0: {
                "analogy": "📖 Think of a library: you walk in with a question (Query), the books have titles on the spine (Keys) and content inside (Values). You haven't asked anything yet — these are just the raw materials.",
                "why": "Three different perspectives on the same input tokens. The model learns to create three *different* representations from each token — one for asking, one for being found, one for providing content.",
                "what_if": "Without separate Q/K/V, every token would use the same representation for searching and being searched. The model couldn't learn 'I need to look for X but I am Y.'",
                "key_insight": "The weight matrices (Wq, Wk, Wv) are learned during training — they transform raw embeddings into useful query/key/value spaces.",
            },
            1: {
                "analogy": "🔍 Think of it like putting on different colored glasses: red glasses help you see what to look for (Query), green glasses help you see what's available (Key), blue glasses help you see the actual content (Value).",
                "why": "Each projection is a learned linear transformation. It rotates and scales the embedding space so that similar things end up close together in the Q space, K space, or V space — but differently in each!",
                "what_if": "Without projection, you'd be comparing raw token embeddings directly. The model would have no flexibility to learn *what aspects* of a token matter for matching vs. providing content.",
                "key_insight": "y = xWᵀ — matrix multiply rotates the vector into a new space. The weight matrix W is learned so that the rotation creates useful representations.",
            },
            2: {
                "analogy": "👥 Imagine a team of 2 analysts (heads), each gets their own slice of the information. Analyst 1 looks at dimensions 0-2, Analyst 2 looks at dimensions 3-5. They each focus on different aspects.",
                "why": "Splitting into heads lets the model attend to *different types of relationships simultaneously*. One head might learn syntactic relationships (subject-verb), another semantic (France-Paris).",
                "what_if": "With 1 head (no split), all attention mixes into one pattern. The model can only learn ONE type of relationship. With multiple heads, it can learn SEVERAL independent relationship types at once.",
                "key_insight": "head_dim = embed_dim / num_heads. We don't lose capacity — we redistribute it. 2 heads of dim 3 can learn 2 different patterns, instead of 1 head of dim 6 learning 1 pattern.",
            },
            3: {
                "analogy": "🔀 Like splitting a conference room into separate breakout rooms. Each head gets its own room so they can have independent conversations without overhearing each other.",
                "why": "We transpose (swap seq ↔ head dims) so that the sequence dimension becomes the 'inner' dimension for each head. This makes each head an independent batch — attention only operates within each head.",
                "what_if": "If we DON'T transpose and keep heads as an inner dimension, the attention computation (QKᵀ) would mix head and sequence indices. Head 0's dim-0 would attend to Head 1's dim-0 — total nonsense!",
                "key_insight": "Shape (H, L, D) means: for each of H heads independently, compute attention over L tokens each with D dimensions. The transpose is what makes the heads independent.",
            },
            4: {
                "analogy": "🎯 The core magic! Each token asks 'which other tokens are relevant to me?' Attention weights are the answer — they're a probability distribution (sum to 1) over all tokens.",
                "why": "QKᵀ measures similarity: how much each query matches each key. Dividing by √d prevents the values from getting too large (which would make softmax spike to one-hot). Softmax normalizes into probabilities. Multiplying by V takes a weighted average of content.",
                "what_if": "Without √d scaling: with large dimensions, QKᵀ values blow up → softmax becomes essentially argmax → attention degenerates to always picking ONE token. No smooth blending! This is called 'dot-product attention scaling.'",
                "key_insight": "softmax(QKᵀ/√d) produces weights where: similar Q-K pairs → high weight. The weighted sum of V means each output token is a blend of all input tokens, weighted by relevance.",
            },
            5: {
                "analogy": "🧩 Like reassembling a jigsaw puzzle. Each head produced its own perspective — now we stitch them all back together so the next layer gets a complete picture.",
                "why": "Concatenation merges the independent head outputs back into the full embedding dimension. Each token now carries the combined insights from ALL relationship types the heads learned.",
                "what_if": "If we didn't merge, we'd have separate head outputs floating around with no way to combine them. The next layer expects embeddings of the original dimension.",
                "key_insight": "The concat is just reshaping: (H, L, D) → (L, H×D) = (L, embed_dim). No computation — just memory rearrangement. The real mixing of head information happens in the output projection.",
            },
            6: {
                "analogy": "🎨 The final touch: a master painter (Wo) mixes the colors from all analysts into a coherent final painting. Each head's perspective gets blended appropriately.",
                "why": "The output projection lets the model learn HOW to combine the different head outputs. Maybe head 1's output is more important for this task, or heads 1 and 2 should be added together. Wo learns this mixing.",
                "what_if": "Without Wo, you'd just concatenate raw head outputs. The model would have no way to learn that 'head 1 is about syntax and should be weighted more for next-word prediction.' Wo provides that flexibility.",
                "key_insight": "The ENTIRE multi-head attention block is: Output = Concat(head₁, head₂, ...) × Wo, where each headᵢ = softmax(QᵢKᵢᵀ/√d) × Vᵢ. Everything is differentiable, so it all learns via backprop.",
            },
        }
        info = explanations.get(self.step, {})
        self.analogy = info.get("analogy", "")
        self.why = info.get("why", "")
        self.what_if = info.get("what_if", "")
        self.key_insight = info.get("key_insight", "")

    @rx.event
    def generate(self):
        self.step = 0
        self.seq_len = 4
        self.embed_dim = 6
        self.num_heads = 2
        self.query_input = make_matrix(self.seq_len, self.embed_dim)
        self.key_input = make_matrix(self.seq_len, self.embed_dim)
        self.value_input = make_matrix(self.seq_len, self.embed_dim)

        self.wq = make_matrix(self.embed_dim, self.embed_dim, -1, 1)
        self.wk = make_matrix(self.embed_dim, self.embed_dim, -1, 1)
        self.wv = make_matrix(self.embed_dim, self.embed_dim, -1, 1)
        self.wo = make_matrix(self.embed_dim, self.embed_dim, -1, 1)

        self.q_proj = []
        self.k_proj = []
        self.v_proj = []
        self.q_heads = []
        self.k_heads = []
        self.v_heads = []
        self.attn_weights = []
        self.attn_output_heads = []
        self.merged = []
        self.output = []
        self._set_explanation()

    @rx.event
    def next_step(self):
        self.step += 1
        head_dim = self.embed_dim // self.num_heads

        if self.step == 1:
            wq_t = transpose(self.wq)
            wk_t = transpose(self.wk)
            wv_t = transpose(self.wv)
            self.q_proj = matmul_2d(self.query_input, wq_t)
            self.k_proj = matmul_2d(self.key_input, wk_t)
            self.v_proj = matmul_2d(self.value_input, wv_t)

        elif self.step == 2:
            self.q_heads = [
                [row[h * head_dim:(h + 1) * head_dim] for h in range(self.num_heads)]
                for row in self.q_proj
            ]
            self.k_heads = [
                [row[h * head_dim:(h + 1) * head_dim] for h in range(self.num_heads)]
                for row in self.k_proj
            ]
            self.v_heads = [
                [row[h * head_dim:(h + 1) * head_dim] for h in range(self.num_heads)]
                for row in self.v_proj
            ]

        elif self.step == 3:
            pass

        elif self.step == 4:
            self.attn_weights = []
            self.attn_output_heads = []
            scale = 1.0 / math.sqrt(head_dim)
            for h in range(self.num_heads):
                q_h = [self.q_heads[s][h] for s in range(self.seq_len)]
                k_h = [self.k_heads[s][h] for s in range(self.seq_len)]
                v_h = [self.v_heads[s][h] for s in range(self.seq_len)]
                k_t = transpose(k_h)
                scores = matmul_2d(q_h, k_t)
                scores = [[s * scale for s in row] for row in scores]
                weights = [softmax(row) for row in scores]
                self.attn_weights.append(weights)
                out_h = matmul_2d(weights, v_h)
                self.attn_output_heads.append(out_h)

        elif self.step == 5:
            self.merged = []
            for s in range(self.seq_len):
                row = []
                for h in range(self.num_heads):
                    row.extend(self.attn_output_heads[h][s])
                self.merged.append(row)

        elif self.step == 6:
            wo_t = transpose(self.wo)
            self.output = matmul_2d(self.merged, wo_t)
            self.step = 6

        self._set_explanation()

    @rx.event
    def prev_step(self):
        if self.step > 0:
            self.step -= 1
            self._set_explanation()

    @rx.event
    def clear_steps(self):
        self.step = 0
        self.q_proj = []
        self.k_proj = []
        self.v_proj = []
        self.q_heads = []
        self.k_heads = []
        self.v_heads = []
        self.attn_weights = []
        self.attn_output_heads = []
        self.merged = []
        self.output = []
        self._set_explanation()

    @rx.var
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @rx.var
    def step_title(self) -> str:
        titles = {
            0: "📥 Input: What are Q, K, V?",
            1: "🔀 Linear Projection: Creating Different Views",
            2: "✂️ Reshape: Splitting into Attention Heads",
            3: "🔄 Transpose: Making Heads Independent",
            4: "🎯 Attention: Where the Magic Happens",
            5: "🔗 Merge: Reassembling Perspectives",
            6: "📤 Output Projection: Final Mix",
        }
        return titles.get(self.step, "")

    @rx.var
    def is_last_step(self) -> bool:
        return self.step >= 6

    @rx.var
    def is_first_step(self) -> bool:
        return self.step <= 0

    @rx.var
    def progress_pct(self) -> int:
        return int(self.step / 6 * 100)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------

def matrix_box(
    data: list[list[float]],
    label: str,
    color: str = "blue",
    decimals: int = 2,
) -> rx.Component:
    return rx.vstack(
        rx.text(label, font_weight="bold", font_size="0.8em", color=f"var(--color-{color}-9)"),
        rx.box(
            rx.foreach(
                data,
                lambda row: rx.hstack(
                    rx.foreach(
                        row,
                        lambda cell: rx.box(
                            rx.text(fmt(cell, decimals), font_size="0.65em"),
                            padding="0.15em 0.3em",
                            min_width="2.5em",
                            text_align="center",
                            border_radius="2px",
                            background=f"var(--color-{color}-3)",
                        ),
                    ),
                    spacing="1",
                ),
            ),
            spacing="1",
        ),
        spacing="1",
        align="center",
    )


def attention_heatmap(weights: list[list[list[float]]]) -> rx.Component:
    return rx.vstack(
        rx.text("Attention Weights (softmax)", font_weight="bold", font_size="0.8em", color="var(--color-orange-9)"),
        rx.hstack(
            rx.foreach(
                weights,
                lambda head_weights: rx.vstack(
                    rx.badge("Head", color_scheme="orange", variant="soft"),
                    rx.foreach(
                        head_weights,
                        lambda row: rx.hstack(
                            rx.foreach(
                                row,
                                lambda cell: rx.box(
                                    rx.text(fmt(cell, 2), font_size="0.6em"),
                                    padding="0.15em 0.25em",
                                    min_width="2.5em",
                                    text_align="center",
                                    border_radius="2px",
                                    background=rx.cond(
                                        cell > 0.5,
                                        "var(--color-orange-8)",
                                        rx.cond(
                                            cell > 0.25,
                                            "var(--color-orange-5)",
                                            "var(--color-orange-2)",
                                        ),
                                    ),
                                    color=rx.cond(cell > 0.5, "white", "black"),
                                ),
                            ),
                            spacing="1",
                        ),
                    ),
                    spacing="1",
                    align="center",
                ),
            ),
            spacing="4",
            align="start",
        ),
        spacing="2",
    )


def explanation_panel() -> rx.Component:
    """The conceptual explanation panel with analogy, why, what-if, insight."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.text("💡 Understanding This Step", font_weight="bold", font_size="1.1em"),
                rx.badge(f"Step {MHAState.step}/6", color_scheme="violet", variant="soft"),
                spacing="3",
                align="center",
            ),
            rx.divider(),
            # Analogy
            rx.hstack(
                rx.text("📖", font_size="1.3em"),
                rx.text(MHAState.analogy, font_size="0.9em", line_height="1.6"),
                spacing="2",
                align="start",
            ),
            rx.divider(margin_y="0.3em"),
            # Why
            rx.hstack(
                rx.text("❓", font_size="1.3em"),
                rx.vstack(
                    rx.text("Why do we do this?", font_weight="bold", font_size="0.9em", color="var(--color-blue-9)"),
                    rx.text(MHAState.why, font_size="0.88em", line_height="1.5", color="gray"),
                    spacing="1",
                ),
                spacing="2",
                align="start",
            ),
            rx.divider(margin_y="0.3em"),
            # What if
            rx.hstack(
                rx.text("⚠️", font_size="1.3em"),
                rx.vstack(
                    rx.text("What if we didn't?", font_weight="bold", font_size="0.9em", color="var(--color-red-9)"),
                    rx.text(MHAState.what_if, font_size="0.88em", line_height="1.5", color="gray"),
                    spacing="1",
                ),
                spacing="2",
                align="start",
            ),
            rx.divider(margin_y="0.3em"),
            # Key insight
            rx.hstack(
                rx.text("🔑", font_size="1.3em"),
                rx.vstack(
                    rx.text("Key Insight", font_weight="bold", font_size="0.9em", color="var(--color-grass-9)"),
                    rx.text(MHAState.key_insight, font_size="0.88em", line_height="1.5", color="gray"),
                    spacing="1",
                ),
                spacing="2",
                align="start",
            ),
            spacing="3",
        ),
        width="100%",
        background="var(--color-slate-2)",
        border_left="4px solid var(--color-violet-8)",
    )


def controls() -> rx.Component:
    return rx.hstack(
        rx.button("Generate New", on_click=MHAState.generate, color_scheme="blue"),
        rx.button("◀ Prev", on_click=MHAState.prev_step, variant="outline", disabled=MHAState.is_first_step),
        rx.button("Next ▶", on_click=MHAState.next_step, color_scheme="grass", disabled=MHAState.is_last_step),
        rx.button("Reset", on_click=MHAState.clear_steps, variant="outline"),
        rx.progress(value=MHAState.progress_pct, width="200px"),
        spacing="3",
        align="center",
        width="100%",
    )


def viz_area() -> rx.Component:
    """The math/visual area that changes per step."""
    return rx.card(
        rx.vstack(
            rx.heading(MHAState.step_title, size="5"),
            rx.hstack(
                rx.badge(f"seq_len={MHAState.seq_len}", color_scheme="blue", variant="outline"),
                rx.badge(f"embed_dim={MHAState.embed_dim}", color_scheme="grass", variant="outline"),
                rx.badge(f"num_heads={MHAState.num_heads}", color_scheme="orange", variant="outline"),
                rx.badge(f"head_dim={MHAState.head_dim}", color_scheme="violet", variant="outline"),
                spacing="2",
            ),

            # Step 0: Inputs
            rx.cond(
                MHAState.step == 0,
                rx.vstack(
                    rx.hstack(
                        matrix_box(MHAState.query_input, "Query (L × E)", "blue"),
                        matrix_box(MHAState.key_input, "Key (L × E)", "green"),
                        matrix_box(MHAState.value_input, "Value (L × E)", "purple"),
                        spacing="6",
                        align="start",
                    ),
                    rx.divider(),
                    rx.hstack(
                        matrix_box(MHAState.wq, "Wq (E × E)", "blue"),
                        matrix_box(MHAState.wk, "Wk (E × E)", "green"),
                        matrix_box(MHAState.wv, "Wv (E × E)", "purple"),
                        matrix_box(MHAState.wo, "Wo (E × E)", "red"),
                        spacing="4",
                        align="start",
                    ),
                    spacing="3",
                ),

                # Step 1: Projections
                rx.cond(
                    MHAState.step == 1,
                    rx.hstack(
                        rx.vstack(
                            matrix_box(MHAState.query_input, "Query", "blue"),
                            rx.text("↓ @ Wqᵀ", font_size="1.3em", align="center"),
                            matrix_box(MHAState.q_proj, "Q projected", "blue"),
                            spacing="2",
                            align="center",
                        ),
                        rx.vstack(
                            matrix_box(MHAState.key_input, "Key", "green"),
                            rx.text("↓ @ Wkᵀ", font_size="1.3em", align="center"),
                            matrix_box(MHAState.k_proj, "K projected", "green"),
                            spacing="2",
                            align="center",
                        ),
                        rx.vstack(
                            matrix_box(MHAState.value_input, "Value", "purple"),
                            rx.text("↓ @ Wvᵀ", font_size="1.3em", align="center"),
                            matrix_box(MHAState.v_proj, "V projected", "purple"),
                            spacing="2",
                            align="center",
                        ),
                        spacing="6",
                        align="start",
                    ),

                    # Step 2: Reshape
                    rx.cond(
                        MHAState.step == 2,
                        rx.vstack(
                            rx.text("Q reshaped: each row split into heads", font_weight="bold", color="var(--color-blue-9)"),
                            rx.box(
                                rx.foreach(
                                    MHAState.q_heads,
                                    lambda row_heads: rx.hstack(
                                        rx.foreach(
                                            row_heads,
                                            lambda head: rx.hstack(
                                                rx.foreach(
                                                    head,
                                                    lambda v: rx.box(
                                                        rx.text(fmt(v, 1), font_size="0.6em"),
                                                        padding="0.1em 0.2em",
                                                        min_width="1.8em",
                                                        text_align="center",
                                                        border_radius="2px",
                                                        background="var(--color-blue-3)",
                                                    ),
                                                ),
                                                spacing="1",
                                            ),
                                        ),
                                        rx.text("|", font_size="0.8em", color="var(--color-grass-9)", font_weight="bold"),
                                        spacing="2",
                                        align="center",
                                    ),
                                ),
                                spacing="1",
                            ),
                            rx.hstack(
                                rx.box(padding="0.2em", background="var(--color-blue-3)", border_radius="2px"),
                                rx.text(" = Head 0", font_size="0.85em"),
                                rx.box(padding="0.2em", background="var(--color-grass-3)", border_radius="2px"),
                                rx.text(" = Head 1", font_size="0.85em"),
                                spacing="2",
                                align="center",
                            ),
                            spacing="3",
                        ),

                        # Step 3: Transpose
                        rx.cond(
                            MHAState.step == 3,
                            rx.vstack(
                                rx.hstack(
                                    rx.vstack(
                                        rx.text("(Seq, Heads, Dim)", font_weight="bold", color="var(--color-blue-9)"),
                                        rx.text("Token 0: [head0 | head1]", font_size="0.85em", color="gray"),
                                        rx.text("Token 1: [head0 | head1]", font_size="0.85em", color="gray"),
                                        rx.text("Token 2: [head0 | head1]", font_size="0.85em", color="gray"),
                                        spacing="1",
                                    ),
                                    rx.text("→", font_size="2em"),
                                    rx.vstack(
                                        rx.text("(Heads, Seq, Dim)", font_weight="bold", color="var(--color-grass-9)"),
                                        rx.text("Head 0: [tok0, tok1, tok2]", font_size="0.85em", color="gray"),
                                        rx.text("Head 1: [tok0, tok1, tok2]", font_size="0.85em", color="gray"),
                                        spacing="1",
                                    ),
                                    spacing="4",
                                    align="center",
                                ),
                                rx.text("Each head now sees all tokens — can attend independently!", font_size="0.9em", color="var(--color-grass-9)", font_weight="bold"),
                                spacing="3",
                            ),

                            # Step 4: Attention
                            rx.cond(
                                MHAState.step == 4,
                                rx.vstack(
                                    attention_heatmap(MHAState.attn_weights),
                                    rx.divider(),
                                    rx.text("Attention Output per head", font_weight="bold", font_size="0.8em", color="var(--color-grass-9)"),
                                    rx.hstack(
                                        rx.foreach(
                                            MHAState.attn_output_heads,
                                            lambda head_out: rx.vstack(
                                                rx.badge("Head", color_scheme="grass", variant="soft"),
                                                rx.foreach(
                                                    head_out,
                                                    lambda row: rx.hstack(
                                                        rx.foreach(
                                                            row,
                                                            lambda cell: rx.box(
                                                                rx.text(fmt(cell, 1), font_size="0.6em"),
                                                                padding="0.1em 0.2em",
                                                                min_width="2em",
                                                                text_align="center",
                                                                border_radius="2px",
                                                                background="var(--color-grass-3)",
                                                            ),
                                                        ),
                                                        spacing="1",
                                                    ),
                                                ),
                                                spacing="1",
                                                align="center",
                                            ),
                                        ),
                                        spacing="4",
                                        align="start",
                                    ),
                                    spacing="3",
                                ),

                                # Step 5: Merge
                                rx.cond(
                                    MHAState.step == 5,
                                    rx.vstack(
                                        matrix_box(MHAState.merged, "Merged (L × E) — all heads concatenated", "violet"),
                                        spacing="2",
                                    ),

                                    # Step 6: Output
                                    rx.vstack(
                                        rx.hstack(
                                            matrix_box(MHAState.merged, "Merged", "violet"),
                                            rx.text("@", font_size="2em", align_self="center"),
                                            matrix_box(MHAState.wo, "Woᵀ", "red"),
                                            rx.text("=", font_size="2em", align_self="center"),
                                            matrix_box(MHAState.output, "Final Output", "red"),
                                            spacing="3",
                                            align="start",
                                        ),
                                        rx.text("🎉 Multi-head attention complete!", font_size="1.1em", color="var(--color-grass-9)", font_weight="bold"),
                                        spacing="2",
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
            spacing="4",
        ),
        width="100%",
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def index() -> rx.Component:
    return rx.container(
        rx.vstack(
            rx.heading("Multi-Head Attention Visualizer", size="8"),
            rx.text(
                "Step through each operation. Don't just see the math — understand WHY each step exists. "
                "Each step has an analogy, a reason, and a 'what if we skipped this?' explanation.",
                color="gray",
                font_size="0.95em",
            ),

            controls(),

            rx.hstack(
                # Left: Math/Visual
                rx.box(viz_area(), width="55%"),
                # Right: Explanation
                rx.box(explanation_panel(), width="45%"),
                spacing="4",
                align="start",
                width="100%",
            ),

            spacing="5",
            align="stretch",
        ),
        max_width="1400px",
        padding="2em",
        on_mount=MHAState.generate,
    )


app = rx.App(theme=rx.theme(appearance="dark", accent_color="violet"))
app.add_page(index, title="Multi-Head Attention Visualizer")
