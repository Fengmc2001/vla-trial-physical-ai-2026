"""Hand-drawn preparation schematic, deliberately not a robotics simulation."""
from PIL import Image, ImageDraw

FIXTURES = {
    "日本語": "青いブロックを左のトレーに、黄色いブロックを右のトレーに入れてください。赤いブロックは動かさないでください。",
    "中文": "把蓝色方块放进左边托盘，黄色方块放进右边托盘。不要移动红色方块。",
    "English": "Put the blue block in the left tray and the yellow block in the right tray. Do not move the red block."
}


def schematic():
    im = Image.new("RGB", (900, 520), "#101b2c")
    d = ImageDraw.Draw(im)
    d.text((24, 18), "CONFIGURATION PREVIEW - NO PHYSICS / NO AI INFERENCE", fill="#f4c66e", font_size=20)
    d.rounded_rectangle((36, 74, 864, 450), radius=16, fill="#dce5ef")
    for x, label in ((100, "left_tray"), (640, "right_tray")):
        d.rounded_rectangle((x, 104, x + 155, 218), radius=12, fill="#a7bcd3", outline="#536e8c", width=3)
        d.text((x + 10, 145), label, fill="#203b56", font_size=20)
    for x, color, label in ((170, "#cf4c55", "red / protected"), (340, "#367cc4", "blue"), (510, "#e7bb43", "yellow"), (680, "#48a88a", "green")):
        d.rounded_rectangle((x, 288, x + 58, 346), radius=8, fill=color)
        d.text((x - 15, 362), label, fill="#253a4e", font_size=16)
    d.ellipse((420, 392, 480, 452), fill="#77889a")
    d.line((450, 421, 400, 248, 470, 186), fill="#506782", width=18)
    d.text((24, 478), "Static layout illustration. Robot execution and model evaluation are pending.", fill="#e4edf7", font_size=18)
    return im


def fixture_plan(language):
    return {"metadata": {"source": "HAND_WRITTEN_SCHEMA_EXAMPLE_NOT_MODEL_OUTPUT", "language": language},
            "plan": {"actions": [{"action": "pick_place", "object_id": "blue_block", "target_id": "left_tray"},
                        {"action": "pick_place", "object_id": "yellow_block", "target_id": "right_tray"}, {"action": "done"}]}}
