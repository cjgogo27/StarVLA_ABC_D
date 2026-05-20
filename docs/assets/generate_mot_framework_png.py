from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(__file__).with_name("mot_enhanced_framework.png")
W, H = 1800, 1050


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf" if bold else "/usr/local/lib/python3.12/dist-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(54, True)
F_SUB = font(26)
F_TAG = font(22, True)
F_HEAD = font(27, True)
F_BODY = font(21)
F_SMALL = font(19)


COL = {
    "bg": "#f6f7fb",
    "panel": "#ffffff",
    "ink": "#172033",
    "muted": "#5b6678",
    "line": "#2f3a4a",
    "soft_line": "#cfd6e3",
    "blue_fill": "#edf4ff",
    "blue": "#4d70b4",
    "mot_fill": "#e8f8f4",
    "mot": "#0b7a75",
    "head_fill": "#fff3e5",
    "head": "#b76e1e",
    "out_fill": "#f2eefb",
    "out": "#6f58b8",
    "note": "#eef2f8",
}


def rounded(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, width: int = 3, r: int = 18) -> None:
    draw.rounded_rectangle(xy, radius=r, fill=fill, outline=outline, width=width)


def shadow_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], fill: str, outline: str, width: int = 3, r: int = 18) -> None:
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle((x1 + 8, y1 + 12, x2 + 8, y2 + 12), radius=r, fill="#dfe4ee")
    rounded(draw, xy, fill, outline, width, r)


def text_center(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], lines: list[tuple[str, ImageFont.FreeTypeFont, str]], gap: int = 8) -> None:
    x1, y1, x2, y2 = xy
    heights = []
    widths = []
    for text, fnt, _ in lines:
        box = draw.textbbox((0, 0), text, font=fnt)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    total_h = sum(heights) + gap * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for (text, fnt, color), tw, th in zip(lines, widths, heights):
        x = x1 + (x2 - x1 - tw) / 2
        draw.text((x, y), text, font=fnt, fill=color)
        y += th + gap


def arrow(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]], color: str = COL["line"], width: int = 4) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 16
    back = 20
    tip = (x2, y2)
    left = (x2 - back * ux + size * 0.5 * px, y2 - back * uy + size * 0.5 * py)
    right = (x2 - back * ux - size * 0.5 * px, y2 - back * uy - size * 0.5 * py)
    draw.polygon([tip, left, right], fill=color)


def main() -> None:
    img = Image.new("RGB", (W, H), COL["bg"])
    draw = ImageDraw.Draw(img)

    draw.text((88, 48), "MOT-Enhanced QwenGR00T Framework", font=F_TITLE, fill=COL["ink"])
    draw.text(
        (90, 122),
        "A compact motion token is appended to Qwen-VL hidden states before the GR00T flow-matching action head.",
        font=F_SUB,
        fill=COL["muted"],
    )

    shadow_box(draw, (70, 170, 1730, 930), COL["panel"], "#d7deea", 2, 22)

    # Input boxes.
    draw.text((112, 214), "INPUTS", font=F_TAG, fill=COL["mot"])
    input_boxes = {
        "rgb": (105, 255, 380, 350),
        "lang": (105, 390, 380, 485),
        "state_hist": (105, 610, 380, 715),
        "action_hist": (105, 755, 380, 860),
    }
    for key, xy in input_boxes.items():
        rounded(draw, xy, "#ffffff", COL["soft_line"], 2, 14)

    text_center(draw, input_boxes["rgb"], [("RGB observation", F_HEAD, COL["ink"]), ("image_t", F_SMALL, COL["muted"])])
    text_center(draw, input_boxes["lang"], [("Instruction", F_HEAD, COL["ink"]), ("language command", F_SMALL, COL["muted"])])
    text_center(draw, input_boxes["state_hist"], [("State history", F_HEAD, COL["ink"]), ("state_{t-K+1:t}", F_SMALL, COL["muted"])])
    text_center(draw, input_boxes["action_hist"], [("Action history", F_HEAD, COL["ink"]), ("optional action_{t-K:t-1}", F_SMALL, COL["muted"])])

    # Main blocks.
    qwen = (510, 320, 850, 480)
    mot = (510, 655, 850, 815)
    motion = (990, 680, 1280, 805)
    fusion = (990, 355, 1300, 485)
    state_now = (990, 220, 1300, 310)
    head = (1410, 345, 1690, 520)
    out = (1410, 665, 1690, 800)

    rounded(draw, qwen, COL["blue_fill"], COL["blue"], 4, 18)
    rounded(draw, mot, COL["mot_fill"], COL["mot"], 4, 18)
    rounded(draw, motion, COL["mot_fill"], COL["mot"], 4, 18)
    rounded(draw, fusion, "#ffffff", "#b8c0cc", 3, 18)
    rounded(draw, state_now, "#ffffff", COL["soft_line"], 2, 14)
    rounded(draw, head, COL["head_fill"], COL["head"], 4, 18)
    rounded(draw, out, COL["out_fill"], COL["out"], 4, 18)

    text_center(draw, qwen, [("Qwen-VL Encoder", F_HEAD, COL["ink"]), ("visual-language grounding", F_BODY, COL["muted"]), ("hidden tokens h_vlm", F_SMALL, COL["muted"])])
    text_center(draw, mot, [("MOT Adapter", F_HEAD, COL["ink"]), ("short-horizon motion", F_BODY, COL["muted"]), ("Transformer over history", F_SMALL, COL["muted"])])
    text_center(draw, motion, [("Motion Token", F_HEAD, COL["ink"]), ("m in Qwen hidden dim", F_SMALL, COL["muted"])])
    text_center(draw, fusion, [("Token Fusion", F_HEAD, COL["ink"]), ("h_aug = concat(h_vlm, m)", F_BODY, COL["muted"])])
    text_center(draw, state_now, [("Current State", F_BODY, COL["ink"]), ("state_t direct path", F_SMALL, COL["muted"])])
    text_center(draw, head, [("GR00T Flow Head", F_HEAD, COL["ink"]), ("flow-matching action model", F_BODY, COL["muted"]), ("cross-attends to h_aug", F_SMALL, COL["muted"])])
    text_center(draw, out, [("Action Chunk", F_HEAD, COL["ink"]), ("a_{t:t+H-1}", F_BODY, COL["muted"]), ("continuous [T, 7]", F_SMALL, COL["muted"])])

    # Arrows.
    arrow(draw, [(380, 302), (510, 360)], COL["line"], 4)
    arrow(draw, [(380, 438), (510, 420)], COL["line"], 4)
    arrow(draw, [(850, 398), (990, 415)], COL["line"], 4)
    arrow(draw, [(1300, 420), (1410, 430)], COL["line"], 5)
    arrow(draw, [(1550, 520), (1550, 665)], COL["line"], 5)

    arrow(draw, [(380, 662), (510, 704)], COL["mot"], 5)
    arrow(draw, [(380, 807), (510, 770)], "#7c8798", 3)
    arrow(draw, [(850, 735), (990, 735)], COL["mot"], 5)
    arrow(draw, [(1135, 680), (1135, 485)], COL["mot"], 5)
    arrow(draw, [(1300, 265), (1390, 345), (1438, 380)], "#687385", 3)

    # Notes and formula.
    rounded(draw, (510, 858, 1280, 916), COL["note"], "#d7deea", 2, 12)
    draw.text(
        (522, 874),
        "MOT path: history -> adapter -> motion token -> action context",
        font=F_BODY,
        fill=COL["ink"],
    )

    rounded(draw, (1390, 844, 1690, 916), "#ffffff", "#d7deea", 2, 12)
    draw.text((1410, 858), "Original modules stay intact;", font=F_SMALL, fill=COL["muted"])
    draw.text((1410, 888), "MOT augments conditioning.", font=F_SMALL, fill=COL["muted"])

    # Bottom caption.
    draw.text(
        (88, 970),
        "Policy form: h_vlm = QwenVL(image_t, instruction),  m = MOT(history),  h_aug = concat(h_vlm, m),  a = GR00T(h_aug, state_t)",
        font=F_BODY,
        fill=COL["muted"],
    )

    img.save(OUT, quality=95)
    print(OUT)


if __name__ == "__main__":
    main()
