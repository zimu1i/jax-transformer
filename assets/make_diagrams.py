"""Generate the PlotNeuralNet-style diagrams in this folder.

Run: python assets/make_diagrams.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"
INK, MUTED, EDGE, ARROW = "#1f2933", "#5f6b7a", "#3a3a3a", "#3f7a78"
K = 0.55  # oblique projection: depth axis drawn up-right at 45 degrees

IDS = "#e4e4e4"
EMB = "#ffe6bf"
LN = "#cdb4db"
SELF = "#bcd3ec"
MASKED = "#8fb3dd"
CROSS = "#f2b8c6"
FFN_UP = "#fdbe7a"
FFN_DOWN = "#ffe6bf"
ENC_OUT = "#bfe3c9"
LOGITS = "#e8836b"
SOFTMAX = "#9b72aa"
HEADS = ["#f6b67a", "#e98a7e", "#8fb3dd", "#b39ad0"]


def mix(c, other, t):
    a = [int(c[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(other[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def lighter(c, t=0.3):
    return mix(c, "#ffffff", t)


def darker(c, t=0.12):
    return mix(c, "#000000", t)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h):
        self.w, self.h, self.parts = w, h, []

    def text(self, x, y, s, size=13, weight="normal", fill=INK, anchor="middle", rotate=None):
        tr = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate is not None else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'fill="{fill}" text-anchor="{anchor}"{tr}>{esc(s)}</text>'
        )

    def poly(self, pts, fill="none", stroke=EDGE, sw=0.9, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.parts.append(
            f'<polygon points="{p}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round"{d}/>'
        )

    def rect(self, x, y, w, h, fill, stroke=EDGE, sw=0.9, rx=0, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def line(self, x1, y1, x2, y2, stroke=EDGE, sw=0.9, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}"{d}/>'
        )

    def arrow(self, pts, color=ARROW, sw=2.6):
        p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        self.parts.append(
            f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}" '
            f'stroke-linejoin="round" marker-end="url(#head)"/>'
        )

    def save(self, name):
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w:.0f} {self.h:.0f}" '
            f'width="{self.w:.0f}" height="{self.h:.0f}" font-family="{FONT}">'
        )
        defs = (
            '<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" '
            f'markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 L3,5 z" fill="{ARROW}"/></marker></defs>'
        )
        bg = f'<rect width="{self.w:.0f}" height="{self.h:.0f}" rx="14" fill="#ffffff"/>'
        (OUT / name).write_text("\n".join([head, defs, bg, *self.parts, "</svg>"]) + "\n")


def cuboid(svg, x, yc, w, h, depth, color, bands=None):
    """Draw a slab whose 3D center sits on the horizontal line yc."""
    ox, oy = depth * K, -depth * K
    yt = yc - h / 2 - oy / 2
    bands = bands or [color]
    bh = h / len(bands)

    svg.poly([(x, yt), (x + w, yt), (x + w + ox, yt + oy), (x + ox, yt + oy)], lighter(bands[0], 0.35))
    for i, c in enumerate(bands):
        y0, y1 = yt + i * bh, yt + (i + 1) * bh
        svg.poly([(x + w, y0), (x + w + ox, y0 + oy), (x + w + ox, y1 + oy), (x + w, y1)],
                 darker(c, 0.1), stroke=darker(c, 0.3), sw=0.5)
        svg.poly([(x, y0), (x + w, y0), (x + w, y1), (x, y1)], c, stroke=darker(c, 0.3), sw=0.5)

    svg.poly([(x, yt), (x + w, yt), (x + w, yt + h), (x, yt + h)])
    svg.poly([(x + w, yt), (x + w + ox, yt + oy), (x + w + ox, yt + h + oy), (x + w, yt + h)])
    svg.poly([(x, yt), (x + ox, yt + oy), (x + w + ox, yt + oy), (x + w, yt)])

    hx, hy = x + ox, yt + oy + h
    for x1, y1, x2, y2 in [(hx, yt + oy, hx, hy), (hx, hy, x + w + ox, hy), (x, yt + h, hx, hy)]:
        svg.line(x1, y1, x2, y2, stroke="#8a8a8a", sw=0.7, dash="4,3")

    return dict(x=x, w=w, h=h, yt=yt, ox=ox, oy=oy, yc=yc)


def side_point(g, u, v):
    """Point on the right face: u in [0,1] goes into depth, v in [0,1] goes down."""
    return (g["x"] + g["w"] + u * g["ox"], g["yt"] + v * g["h"] + u * g["oy"])


def side_matrix(svg, g, values, color, masked_fill="#9e9e9e"):
    n = len(values)
    for r in range(n):
        for c in range(n):
            pts = [side_point(g, c / n, r / n), side_point(g, (c + 1) / n, r / n),
                   side_point(g, (c + 1) / n, (r + 1) / n), side_point(g, c / n, (r + 1) / n)]
            v = values[r][c]
            fill = masked_fill if v is None else mix("#ffffff", color, 0.3 + 0.7 * v)
            svg.poly(pts, fill, stroke="#ffffff", sw=0.8)
    svg.poly([side_point(g, 0, 0), side_point(g, 1, 0), side_point(g, 1, 1), side_point(g, 0, 1)])


def group(svg, x, yc, slabs, depth, caption=(), caption_y=None, diag=None, boxed=None):
    geoms, cx = [], x
    for sl in slabs:
        g = cuboid(svg, cx, yc, sl["w"], sl["h"], depth, sl["color"], sl.get("bands"))
        if "side" in sl:
            sl["side"](svg, g)
        geoms.append(g)
        cx += sl["w"]

    bottom = max(g["yt"] + g["h"] for g in geoms)
    for sl, g in zip(slabs, geoms):
        if sl.get("label"):
            svg.text(g["x"] + g["w"] / 2, bottom + 18, sl["label"], 12)

    last = geoms[-1]
    if diag:
        lowest = [g for g in geoms if g["yt"] + g["h"] == bottom][-1]
        mx = lowest["x"] + lowest["w"] + lowest["ox"] / 2 + 10
        my = bottom + lowest["oy"] / 2 + 10
        svg.text(mx, my, diag, 13, rotate=-45)

    cy = caption_y if caption_y is not None else bottom + 44
    for i, line in enumerate(caption):
        if i == 0:
            svg.text((x + cx) / 2, cy, line, 13, "bold")
        else:
            svg.text((x + cx) / 2, cy + 16 * i, line, 12, fill=MUTED)

    if boxed:
        top = min(g["yt"] + g["oy"] for g in geoms) - 12
        right = cx + last["ox"] + 10
        svg.rect(x - 10, top, right - x + 10, bottom + 26 - top, "none", stroke="#9aa5b1",
                 sw=1.2, rx=10, dash="6,4")
        svg.text(right - 8, top + 18, boxed, 14, "bold", fill=MUTED, anchor="end")

    return dict(x0=x, x1=cx, geoms=geoms, right=(cx + last["ox"] / 2, yc), end=cx + last["ox"])


def legend(svg, x, y, items, per_row):
    for i, (color, name) in enumerate(items):
        row, col = divmod(i, per_row)
        lx, ly = x + col * 190, y + row * 26
        svg.rect(lx, ly - 11, 14, 14, color, rx=2)
        svg.text(lx + 22, ly + 1, name, 12, anchor="start")


def full_architecture():
    D, W, WL = 110, 24, 9
    HD, HFF, HV = 110, 190, 150
    ye, yd = 250, 625
    G = 46
    ox = D * K

    s = Svg(0, 910)

    def ln():
        return dict(w=WL, h=HD, color=LN)

    s.text(40, 112, "ENCODER", 15, "bold", fill="#1e40af", anchor="start")
    cap_e = ye + HFF / 2 + ox / 2 + 44
    x = 40
    g1 = group(s, x, ye, [dict(w=W, h=14, color=IDS)], D,
               ["source ids", '"hola"'], cap_e, diag="S")
    x = g1["end"] + G
    g2 = group(s, x, ye, [dict(w=W, h=HD, color=EMB, label="64")], D,
               ["embedding", "token + position"], cap_e, diag="S")
    x = g2["end"] + G
    g3 = group(s, x, ye, [ln(), dict(w=W, h=HD, color=SELF, label="64"), ln(),
                          dict(w=W, h=HFF, color=FFN_UP, label="256"),
                          dict(w=W, h=HD, color=FFN_DOWN, label="64")], D,
               ["encoder block", "self-attention, FFN"], cap_e, diag="S", boxed="x N")
    x = g3["end"] + G
    g4 = group(s, x, ye, [dict(w=W, h=HD, color=ENC_OUT, label="64")], D,
               ["encoder output", "(S, 64)"], cap_e, diag="S")
    for a, b in [(g1, g2), (g2, g3), (g3, g4)]:
        s.arrow([a["right"], (b["x0"] - 2, ye)])

    # Line the decoder up so its cross-attention block sits just right of encoder_out.
    kv_x = g4["right"][0] + 80
    cross_offset = 2 * (W + ox + G) + (WL + W + WL) + W / 2 + ox / 2
    x = kv_x - cross_offset

    s.text(x, 488, "DECODER", 15, "bold", fill="#9d174d", anchor="start")
    cap_d = yd + HFF / 2 + ox / 2 + 44
    d1 = group(s, x, yd, [dict(w=W, h=14, color=IDS)], D,
               ["target ids", '"<bos>hello"'], cap_d, diag="T")
    x = d1["end"] + G
    d2 = group(s, x, yd, [dict(w=W, h=HD, color=EMB, label="64")], D,
               ["embedding", "token + position"], cap_d, diag="T")
    x = d2["end"] + G
    d3 = group(s, x, yd, [ln(), dict(w=W, h=HD, color=MASKED, label="64"), ln(),
                          dict(w=W, h=HD, color=CROSS, label="64"), ln(),
                          dict(w=W, h=HFF, color=FFN_UP, label="256"),
                          dict(w=W, h=HD, color=FFN_DOWN, label="64")], D,
               ["decoder block", "masked attn, cross-attn, FFN"], cap_d, diag="T", boxed="x N")
    x = d3["end"] + G
    d4 = group(s, x, yd, [ln(), dict(w=W, h=HV, color=LOGITS, label="V")], D,
               ["logits", "x @ embedding.T"], cap_d, diag="T")
    x = d4["end"] + G
    d5 = group(s, x, yd, [dict(w=W, h=HV, color=SOFTMAX, label="V")], D,
               ["softmax", "next-char probs"], cap_d, diag="T")
    for a, b in [(d1, d2), (d2, d3), (d3, d4), (d4, d5)]:
        s.arrow([a["right"], (b["x0"] - 2, yd)])

    cross = d3["geoms"][3]
    top = (cross["x"] + cross["w"] / 2 + cross["ox"] / 2, cross["yc"] - cross["h"] / 2)
    s.arrow([g4["right"], (top[0], ye), (top[0], top[1] - 4)])
    s.text(top[0] + 12, 420, "keys and values", 13, "bold", fill=ARROW, anchor="start")
    s.text(top[0] + 12, 436, "for cross-attention in", 12, fill=MUTED, anchor="start")
    s.text(top[0] + 12, 452, "every decoder block", 12, fill=MUTED, anchor="start")

    legend(s, 40, 868, [
        (EMB, "embedding"), (LN, "LayerNorm"), (SELF, "self-attention"),
        (MASKED, "masked self-attention"), (CROSS, "cross-attention"),
        (FFN_UP, "FFN expand (d_ff)"), (ENC_OUT, "encoder output"),
        (LOGITS, "logits"), (SOFTMAX, "softmax"), (IDS, "token ids"),
    ], per_row=5)
    s.w = max(d5["end"] + 50, 1000)
    s.text(s.w / 2, 40, "Encoder-Decoder Transformer: tensors flowing through the network", 20, "bold")
    s.text(s.w / 2, 64, "One example (batch dim omitted). Numbers under a block = feature size. "
                        "Diagonal label = sequence length. Not to scale.", 13, fill=MUTED)
    s.save("full_architecture.svg")


def causal_weights(n, seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(n, n))
    out = []
    for r in range(n):
        row = np.exp(scores[r, : r + 1])
        row /= row.sum()
        out.append([float(row[c]) if c <= r else None for c in range(n)])
    return out


def multihead_attention():
    D, W, HM, HS = 104, 24, 128, 104
    yc, G = 300, 40
    ox = D * K
    n = 5

    s = Svg(0, 540)

    cap = yc + HM / 2 + ox / 2 + 44
    x = 40
    g1 = group(s, x, yc, [dict(w=W, h=HM, color=EMB, label="64")], D,
               ["input x", "(S, 64)"], cap, diag="S")
    x = g1["end"] + G
    g2 = group(s, x, yc, [dict(w=W, h=HM, color=c, bands=HEADS, label=l)
                          for c, l in zip(HEADS, "QKV")], D,
               ["Q, K, V projections", "4 heads of 16 each"], cap, diag="S")
    x = g2["end"] + G

    def score_side(seed):
        rng = np.random.default_rng(seed)
        vals = [[float(rng.uniform(0.2, 1.0)) if c <= r else None for c in range(n)] for r in range(n)]
        return lambda svg, g: side_matrix(svg, g, vals, HEADS[3])

    g3 = group(s, x, yc, [dict(w=12, h=HS, color=c) for c in HEADS[:-1]]
               + [dict(w=12, h=HS, color=HEADS[-1], side=score_side(1))], D,
               ["scores per head", "QKᵀ / √16 + mask"], cap, diag="S")
    x = g3["end"] + G
    weights = causal_weights(n, 2)
    g4 = group(s, x, yc, [dict(w=12, h=HS, color=c) for c in HEADS[:-1]]
               + [dict(w=12, h=HS, color=HEADS[-1],
                       side=lambda svg, g: side_matrix(svg, g, weights, SOFTMAX))], D,
               ["softmax", "rows sum to 1"], cap, diag="S")
    x = g4["end"] + G
    g5 = group(s, x, yc, [dict(w=W, h=HM / 4, color=c, label="16") for c in HEADS], D,
               ["weights @ V", "(4, S, 16)"], cap, diag="S")
    x = g5["end"] + G
    g6 = group(s, x, yc, [dict(w=W, h=HM, color=HEADS[0], bands=HEADS, label="64")], D,
               ["concat heads", "(S, 64)"], cap, diag="S")
    x = g6["end"] + G
    g7 = group(s, x, yc, [dict(w=W, h=HM, color=ENC_OUT, label="64")], D,
               ["output @ W_O", "mixes the heads"], cap, diag="S")

    for a, b in [(g1, g2), (g3, g4), (g4, g5), (g5, g6), (g6, g7)]:
        s.arrow([a["right"], (b["x0"] - 2, yc)])
    s.arrow([g2["right"], (g3["x0"] - 2, yc)])

    v = g2["geoms"][2]
    v_top = (v["x"] + v["w"] / 2 + v["ox"] / 2, yc - HM / 2)
    tgt = g5["geoms"][0]
    t_top = (tgt["x"] + tgt["w"] / 2 + tgt["ox"] / 2, yc - HM / 8)
    route_y = 108
    s.arrow([v_top, (v_top[0], route_y), (t_top[0], route_y), (t_top[0], t_top[1] - 4)])
    s.text((v_top[0] + t_top[0]) / 2, route_y - 8,
           "Q and K build the scores. V skips ahead and gets weighted by them.",
           12, "bold", fill=ARROW)

    s.w = g7["end"] + 40
    s.text(s.w / 2, 40, "Multi-Head Attention: splitting d_model into heads", 20, "bold")
    s.text(s.w / 2, 64, "d_model = 64, h = 4 heads, head_dim = 16. Each color is one head. "
                        "Diagonal label = sequence length S.", 13, fill=MUTED)
    s.text(s.w / 2, cap + 46, "Gray cells on the S x S score matrix are future positions masked to -inf "
                              "(decoder only). The encoder leaves every cell visible.", 12, fill=MUTED)
    s.save("multihead_attention.svg")


def scaled_dot_product_attention():
    tokens = ["<bos>", "h", "e", "l", "l"]
    n, dk, cell = 5, 4, 30
    rng = np.random.default_rng(7)
    Q = np.round(rng.normal(size=(n, dk)), 1)
    Kt = np.round(rng.normal(size=(n, dk)), 1).T
    V = np.round(rng.normal(size=(n, dk)), 1)
    scores = Q @ Kt / np.sqrt(dk)
    masked = np.where(np.tril(np.ones((n, n))) == 1, scores, -np.inf)
    weights = np.exp(masked - masked.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    out = weights @ V

    s = Svg(900, 720)
    s.text(450, 40, "Scaled Dot-Product Attention, one head", 20, "bold")
    s.text(450, 64, "5 tokens, d_k = 4, real numbers from a random example. "
                    "Causal (decoder) mask shown.", 13, fill=MUTED)

    def matrix(x, y, vals, color, title, shape, row_labels=False, col_labels=False, fmt="{:.1f}",
               scale=None):
        rows, cols = vals.shape
        finite = vals[np.isfinite(vals)]
        lo, hi = scale or (finite.min(), finite.max())
        for r in range(rows):
            for c in range(cols):
                v = vals[r, c]
                cx, cy = x + c * cell, y + r * cell
                if not np.isfinite(v):
                    s.rect(cx, cy, cell, cell, "#e5e5e5", stroke="#ffffff", sw=1)
                    s.text(cx + cell / 2, cy + cell / 2 + 4, "-inf", 9, fill="#8a8a8a")
                    continue
                t = (v - lo) / (hi - lo) if hi > lo else 0.5
                fill = mix("#ffffff", color, 0.12 + 0.75 * t)
                s.rect(cx, cy, cell, cell, fill, stroke="#ffffff", sw=1)
                label = fmt.format(v)
                if label.lstrip("-").strip("0.") == "":
                    label = label.lstrip("-")
                s.text(cx + cell / 2, cy + cell / 2 + 4, label, 10,
                       fill="#ffffff" if t > 0.7 else INK)
        s.rect(x, y, cols * cell, rows * cell, "none", stroke=EDGE, sw=1)
        s.text(x + cols * cell / 2, y - (30 if col_labels else 12), title, 13, "bold")
        s.text(x + cols * cell / 2, y + rows * cell + 18, shape, 12, fill=MUTED)
        if row_labels:
            for r, tok in enumerate(tokens):
                s.text(x - 6, y + r * cell + cell / 2 + 4, tok, 11, fill=MUTED, anchor="end")
        if col_labels:
            for c, tok in enumerate(tokens):
                s.text(x + c * cell + cell / 2, y - 8, tok, 10, fill=MUTED)

    def op(x, y, sym):
        s.text(x, y + 8, sym, 24, "bold", fill=MUTED)

    y1 = 150
    mid1 = y1 + n * cell / 2
    matrix(60, y1, Q, "#3b82f6", "Q", "(S, d_k)", row_labels=True)
    op(200, mid1, "×")
    matrix(230, y1 + cell / 2, Kt, "#ec4899", "Kᵀ", "(d_k, S)", col_labels=True)
    op(405, mid1, "/ √4")
    matrix(450, y1, scores, "#f59e0b", "scores", "(S, S)", col_labels=True)
    s.arrow([(608, mid1), (668, mid1)])
    s.text(638, mid1 - 10, "+ mask", 12, "bold", fill=ARROW)
    matrix(676, y1, masked, "#f59e0b", "masked scores", "future set to -inf", col_labels=True,
           scale=(scores.min(), scores.max()))

    y2 = 470
    mid2 = y2 + n * cell / 2
    s.arrow([(676 + n * cell / 2, y1 + n * cell + 30), (676 + n * cell / 2, 370),
             (60 + n * cell / 2, 370), (60 + n * cell / 2, y2 - 44)])
    s.text(400, 362, "softmax over each row", 13, "bold", fill=ARROW)
    matrix(60, y2, weights, "#9b72aa", "attention weights", "(S, S)", row_labels=True,
           col_labels=True, fmt="{:.2f}", scale=(0, 1))
    s.text(60 + n * cell + 22, y2 - 8, "sum", 10, fill=MUTED)
    for r in range(n):
        s.text(60 + n * cell + 22, y2 + r * cell + cell / 2 + 4, "1.00", 10, fill=MUTED)
    op(268, mid2, "×")
    matrix(300, y2, V, "#10b981", "V", "(S, d_k)")
    op(436, mid2, "=")
    matrix(470, y2, out, "#22c55e", "output", "(S, d_k)")

    s.text(640, y2 + 30, "Row 1 (<bos>) can only see itself,", 12, fill=MUTED, anchor="start")
    s.text(640, y2 + 48, "so its weight is 1.00. Row 5 sees", 12, fill=MUTED, anchor="start")
    s.text(640, y2 + 66, "all five tokens. Each output row is", 12, fill=MUTED, anchor="start")
    s.text(640, y2 + 84, "a weighted average of the V rows", 12, fill=MUTED, anchor="start")
    s.text(640, y2 + 102, "it is allowed to look at.", 12, fill=MUTED, anchor="start")

    s.text(450, 690, "Attention(Q, K, V) = softmax(QKᵀ / √d_k) V", 15, "bold")
    s.save("scaled_dot_product_attention.svg")


if __name__ == "__main__":
    full_architecture()
    multihead_attention()
    scaled_dot_product_attention()
