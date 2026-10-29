#!/usr/bin/env python3
"""Create a monochrome Japanese academic report from verified public run evidence.

This document-only tool neither starts inference nor imports the simulator.
The previous presentation-style report is preserved in its original location.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

from PIL import Image as PILImage, ImageDraw, ImageFont
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable, Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
    Spacer, Table, TableStyle,
)

from build_submission_report import load_evidence, read_json, require


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BLACK, WHITE = colors.black, colors.white
REPO = "https://github.com/Fengmc2001/vla-trial-physical-ai-2026"
VIDEO_URL = "https://www.youtube.com/watch?v=hS34eNkm5VM"
FONT = "AcademicSerif"
BOLD = "AcademicSerifBold"
LATIN = "AcademicLatin"
SYMBOL = "AcademicSymbol"
W = A4[0] - 48 * 72 / 25.4


def register_fonts(path: Path | None, index: int | None) -> None:
    candidates = [(path, index or 0)] if path else [
        (Path("/System/Library/Fonts/Supplemental/Songti.ttc"), 7),
        (Path("/usr/share/fonts/truetype/ipaexm.ttf"), 0),
        (Path("/usr/share/fonts/truetype/ipaexfont-mincho/ipaexm.ttf"), 0),
    ]
    found = next(((p, i) for p, i in candidates if p and p.is_file()), None)
    require(found is not None, "Supply --font with a Japanese TrueType serif font")
    p, i = found
    pdfmetrics.registerFont(TTFont(FONT, str(p), subfontIndex=i))
    bold_index = 2 if p.name == "Songti.ttc" else i
    pdfmetrics.registerFont(TTFont(BOLD, str(p), subfontIndex=bold_index))
    latin_path = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
    if latin_path.is_file():
        pdfmetrics.registerFont(TTFont(LATIN, str(latin_path)))
    else:
        pdfmetrics.registerFont(TTFont(LATIN, str(p), subfontIndex=i))
    symbols = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    pdfmetrics.registerFont(TTFont(SYMBOL, str(symbols if symbols.is_file() else p), subfontIndex=0 if symbols.is_file() else i))


def projected(position: list[float]) -> tuple[float, float]:
    """Project the recorded world coordinates with the fixed, recorded camera."""
    import numpy as np
    camera = np.array([1.15, -1.25, 1.0])
    forward = np.array([0.35, 0.0, 0.30]) - camera
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = np.asarray(position) - camera
    focal = 720 / (2 * math.tan(math.radians(50) / 2))
    return (480 + focal * float(delta @ right) / float(delta @ forward),
            360 - focal * float(delta @ up) / float(delta @ forward))


def annotated_gray(source: Path, output: Path, positions: dict) -> None:
    with PILImage.open(source) as original:
        require(original.size == (960, 720), "Review annotation projection for a different camera resolution")
        gray = original.convert("L")
    draw = ImageDraw.Draw(gray)
    font = ImageFont.load_default(size=30)
    offsets = {"blue_block": (-90, -43), "yellow_block": (65, -12), "red_block": (-95, 32)}
    for object_id, letter in [("blue_block", "b"), ("yellow_block", "y"), ("red_block", "r")]:
        x, y = projected(positions[object_id])
        dx, dy = offsets[object_id]
        cx, cy = x + dx, y + dy
        draw.line([(x, y), (cx, cy)], fill=255, width=6)
        draw.line([(x, y), (cx, cy)], fill=0, width=2)
        draw.rectangle([cx - 17, cy - 20, cx + 18, cy + 20], fill=255, outline=0, width=2)
        draw.text((cx, cy), letter, fill=0, font=font, anchor="mm")
    for letter, point, offset in [("A", [.5, .3, .0225], (65, -62)), ("B", [.5, -.3, .0225], (-90, 55))]:
        x, y = projected(point)
        cx, cy = x + offset[0], y + offset[1]
        draw.line([(x, y), (cx, cy)], fill=255, width=6)
        draw.line([(x, y), (cx, cy)], fill=0, width=2)
        draw.rectangle([cx - 18, cy - 20, cx + 19, cy + 20], fill=255, outline=0, width=2)
        draw.text((cx, cy), letter, fill=0, font=font, anchor="mm")
    gray.save(output)


def make_frames(data, directory: Path) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    trajectory = read_json(data.run / "trajectory.json")
    initial = read_json(data.run / "initial.json")
    env = data.environment
    offset = initial["simulation_seconds"] - 1 / 10
    require(env["simulation_settings"]["dt_seconds"] == .01, "Review video time mapping")
    frames = [
        (0.0, "(a) 初期状態", "initial.png"),
        (39.5, "(b) 初期分類の完了", None),
        (58.4, "(c) 青をAの外へ押し出した状態", None),
        (80.2, "(d) 黄を青の上に積んだ状態", None),
        (96.3, "(e) 2個を同時に把持して持ち上げた状態", None),
        (120.0, "(f) 積層を保ってBへ配置した最終状態", "final.png"),
    ]
    ffmpeg = shutil.which("ffmpeg")
    require(ffmpeg is not None, "ffmpeg is required for genuine run-frame extraction")
    manifest = []
    for n, (t, caption, original_name) in enumerate(frames):
        source = data.run / original_name if original_name else directory / f"source-{n}.png"
        if not original_name:
            subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-ss", f"{t + offset:.4f}",
                            "-i", str(data.run / "simulation.mp4"), "-frames:v", "1", "-threads", "1", str(source)],
                           check=True, capture_output=True, timeout=30)
        nearest = min(trajectory, key=lambda row: abs(row["elapsed_s"] - t))
        target = directory / f"frame-{n}.png"
        annotated_gray(source, target, nearest["positions"])
        manifest.append({"image": target.name, "caption": caption, "task_elapsed_s": t,
                         "position_sample_elapsed_s": nearest["elapsed_s"],
                         "raw_video_pts_s": None if original_name else round(t + offset, 4),
                         "original_source": original_name or "simulation.mp4",
                         "image_mode": "L", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    (directory / "figure-sources.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=BLACK, strokeWidth=.7))
    angle = math.atan2(y2-y1, x2-x1)
    for extra in (-.45, .45):
        d.add(Line(x2, y2, x2-5*math.cos(angle+extra), y2-5*math.sin(angle+extra), strokeColor=BLACK, strokeWidth=.7))


def architecture() -> Drawing:
    d = Drawing(W, 118)
    labels = [("画像と混合言語指示", 0), ("QwenのJSON計画", 157), ("許可操作と対象の検査", 314)]
    for label, x in labels:
        d.add(Rect(x, 77, 145, 30, fillColor=WHITE, strokeColor=BLACK, strokeWidth=.7))
        d.add(String(x+72.5, 88, label, textAnchor="middle", fontName=FONT, fontSize=9))
    arrow(d,145,92,157,92); arrow(d,302,92,314,92)
    for label, x in [("動作プリミティブ",314),("Genesis: IK / 関節制御",157),("物理動作と位置履歴",0)]:
        d.add(Rect(x, 20, 145, 30, fillColor=WHITE, strokeColor=BLACK, strokeWidth=.7))
        d.add(String(x+72.5,31,label,textAnchor="middle",fontName=FONT,fontSize=9))
    arrow(d,386,77,386,50); arrow(d,314,35,302,35); arrow(d,157,35,145,35)
    d.add(String(W/2,2,"シミュレータの物体真値座標を低層制御に使用",fontName=FONT,fontSize=8.5,textAnchor="middle"))
    return d


def plot_axes(d, x, y, w, h, xlim, ylim, xticks, yticks, xlabel, ylabel):
    def transform(t, value):
        return (x+(t-xlim[0])/(xlim[1]-xlim[0])*w,
                y+(value-ylim[0])/(ylim[1]-ylim[0])*h)
    d.add(Line(x,y,x+w,y,strokeColor=BLACK,strokeWidth=.6))
    d.add(Line(x,y,x,y+h,strokeColor=BLACK,strokeWidth=.6))
    for tick in xticks:
        px,_=transform(tick,ylim[0]);d.add(Line(px,y,px,y-3,strokeColor=BLACK,strokeWidth=.5))
        d.add(String(px,y-14,f"{tick:g}",fontName=LATIN,fontSize=8,textAnchor="middle"))
    for tick in yticks:
        _,py=transform(xlim[0],tick);d.add(Line(x-3,py,x,py,strokeColor=BLACK,strokeWidth=.5))
        d.add(String(x-6,py-2.5,f"{tick:g}",fontName=LATIN,fontSize=8,textAnchor="end"))
    d.add(String(x+w/2,y-29,xlabel,fontName=LATIN,fontSize=8.5,textAnchor="middle"))
    d.add(String(x,y+h+10,ylabel,fontName=LATIN,fontSize=9))
    return transform


def draw_trace(d, pairs, transform, *, dashed=False, width=.8):
    points=[]
    for t,value in pairs:
        px,py=transform(t,value);points.extend((px,py))
    d.add(PolyLine(points,strokeColor=BLACK,strokeWidth=width,strokeDashArray=[4,2] if dashed else None))


def result_plot(trajectory: list, start: float, end: float) -> Drawing:
    d=Drawing(W,286)
    top=plot_axes(d,42,168,W-57,83,(0,120),(0,260),[0,20,40,60,80,100,120],[0,100,200],
                  "Task elapsed time (s)","(a) Block-center height (mm)")
    for object_id,dashed in [("blue_block",False),("yellow_block",True)]:
        draw_trace(d,[(r["elapsed_s"],r["positions"][object_id][2]*1000) for r in trajectory],top,dashed=dashed)
    for t in (19.8,39.6,58.5,80.25,119.0):
        x,y=top(t,0);_,y2=top(t,260)
        d.add(Line(x,y,x,y2,strokeColor=BLACK,strokeWidth=.3,strokeDashArray=[1,3]))
    d.add(Line(290,272,310,272,strokeColor=BLACK,strokeWidth=.8))
    d.add(String(315,269,"blue",fontName=LATIN,fontSize=9))
    d.add(Line(355,272,375,272,strokeColor=BLACK,strokeWidth=.8,strokeDashArray=[4,2]))
    d.add(String(380,269,"yellow",fontName=LATIN,fontSize=9))
    samples=[r for r in trajectory if start-1e-9 <= r["elapsed_s"] <= end+1e-9]
    metrics=[]
    for r in samples:
        b,y=r["positions"]["blue_block"],r["positions"]["yellow_block"]
        metrics.append((r["elapsed_s"],math.hypot(y[0]-b[0],y[1]-b[1])*1000,(y[2]-b[2])*1000))
    left=plot_axes(d,42,40,170,75,(80.25,119),(0,.3),[85,100,115],[0,.1,.2,.3],
                   "Task elapsed time (s)","(b) Horizontal offset (mm)")
    right=plot_axes(d,288,40,156,75,(80.25,119),(24.9,25.0),[85,100,115],[24.9,24.95,25.0],
                    "Task elapsed time (s)","(c) Vertical separation (mm)")
    draw_trace(d,[(t,xy) for t,xy,z in metrics],left)
    draw_trace(d,[(t,z) for t,xy,z in metrics],right)
    return d


class Rule(Flowable):
    def __init__(self,width):super().__init__();self.width=width;self.height=6
    def draw(self):
        self.canv.setStrokeColor(BLACK);self.canv.setLineWidth(.6);self.canv.line(0,3,self.width,3)


def build(data, output: Path, account: str, figures: Path, manuscript: Path) -> dict:
    frames=make_frames(data,figures)
    trajectory=read_json(data.run/"trajectory.json")
    ev=data.evaluation; milestones=ev["milestone_results"]; contract=ev["contract"]
    transport=milestones["stack_preserved_during_transport"]
    carried=[r for r in trajectory if transport["started_s"]-1e-9 <= r["elapsed_s"] <= transport["ended_s"]+1e-9]
    require(len(carried)==transport["sampled_instants"],"Transport sample count differs")
    gaps=[(r["positions"]["yellow_block"][2]-r["positions"]["blue_block"][2])*1000 for r in carried]
    offsets=[math.dist(r["positions"]["yellow_block"][:2],r["positions"]["blue_block"][:2])*1000 for r in carried]
    final=trajectory[-1]["positions"];b,y=final["blue_block"],final["yellow_block"]
    final_xy=math.dist(b[:2],y[:2])*1000;final_gap=(y[2]-b[2])*1000
    red_m=ev["protected_results"]["red_block"]["maximum_displacement_m"]
    controls=read_json(data.run/"control-events.json")
    lift=next(r for r in controls if r["stage"]=="stack_pair_lift_measured")["lift_heights_m"]
    max_sample_gap=max(trajectory[i+1]["elapsed_s"]-trajectory[i]["elapsed_s"] for i in range(len(trajectory)-1))
    styles={
        "body":ParagraphStyle("Body",fontName=FONT,fontSize=10.2,leading=16.2,spaceAfter=7,wordWrap="CJK",textColor=BLACK,alignment=TA_JUSTIFY),
        "small":ParagraphStyle("Small",fontName=FONT,fontSize=8.8,leading=13.2,spaceAfter=5,wordWrap="CJK",textColor=BLACK),
        "title":ParagraphStyle("Title",fontName=BOLD,fontSize=16.2,leading=24,alignment=TA_CENTER,spaceAfter=10,wordWrap="CJK",textColor=BLACK),
        "meta":ParagraphStyle("Meta",fontName=FONT,fontSize=9.3,leading=14,alignment=TA_CENTER,spaceAfter=8,textColor=BLACK),
        "h1":ParagraphStyle("H1",fontName=BOLD,fontSize=11.4,leading=18,spaceBefore=9,spaceAfter=6,textColor=BLACK),
        "h2":ParagraphStyle("H2",fontName=BOLD,fontSize=10.4,leading=16,spaceBefore=6,spaceAfter=4,textColor=BLACK),
        "caption":ParagraphStyle("Caption",fontName=FONT,fontSize=9,leading=13.3,spaceBefore=5,spaceAfter=8,wordWrap="CJK",textColor=BLACK),
        "cell":ParagraphStyle("Cell",fontName=FONT,fontSize=9.0,leading=13.1,spaceAfter=0,wordWrap="CJK",textColor=BLACK),
    }
    story=[];text_log=[];all_text=[]
    def para(text,kind="body",*,record=True):
        all_text.append(text)
        if record:text_log.append(text)
        markup=escape(text).replace("\n","<br/>").replace("・",f'<font name="{SYMBOL}">・</font>')
        return Paragraph(markup,styles[kind])
    def add(text,kind="body"):
        story.append(para(text,kind))
    def heading(text,level=1):
        text_log.append("#"*(level+1)+" "+text);story.append(para(text,"h1" if level==1 else "h2",record=False))
    def table(rows,widths):
        cells=[[para(str(c),"cell",record=False) for c in row] for row in rows]
        t=Table(cells,colWidths=widths,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([
            ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),5),("RIGHTPADDING",(0,0),(-1,-1),5),
            ("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),
            ("LINEABOVE",(0,0),(-1,0),.7,BLACK),("LINEBELOW",(0,0),(-1,0),.4,BLACK),
            ("LINEBELOW",(0,-1),(-1,-1),.7,BLACK),
        ]));story.append(t);story.append(Spacer(1,6))
        text_log.append("\n".join(" | ".join(str(c) for c in row) for row in rows))
    def page():story.append(PageBreak());text_log.append("\n---\n")

    # Page 1: a conventional paper opening, task definition and system figure.
    add("視覚言語モデルによる混合言語指示の行動計画と\n物理シミュレーションにおける多段階ブロック操作","title")
    add(f"Physical AI 応用1講座　最終課題レポート\nOmniCampusアカウント名：{account}　　2026年10月29日","meta")
    heading("要旨")
    add("日本語・英語・中国語が混在する一つの指示から，視覚言語モデルQwen3-VL-8B-Instructが順序付きの行動計画を生成し，Genesis内のFranka Pandaが分類，押し出し，積み重ね，2物体の同時搬送を実行するシステムを構築した。モデルは対象と操作順序を選択し，低層制御はシミュレータの物体真値座標と逆運動学を用いる。固定条件で行った1回の正式試行では，5操作に対する7項目の段階評価を満たした。搬送中の最大水平ずれは0.266 mmであった。本結果は単一事例の実装検証であり，一般的な成功率や実機精度を示さない。")
    add("キーワード：視覚言語モデル，行動計画，ロボット操作，逆運動学，物理シミュレーション","small")
    heading("1. はじめに")
    add("本課題では，自然言語で与えた作業をロボットの物理動作につなぐことを目的とする。単純な分類に加え，同じ物体への再操作，別の物体を支持物とする積み重ね，操作してはいけない物体の保護を含む課題を設定した。検証の焦点は，混合言語指示から生成した計画と実際の動作が対応し，各段階の達成を記録に基づいて確認できるかに置く。")
    heading("2. 課題設定とシステム構成")
    add("場面には，一辺25 mmの青・黄・赤の立方体とトレーA・Bを配置した。以下の原文をモデルへ入力し，青をAへ，黄をBへ移動した後，青をAの外へ押し出し，黄を青の上に積み，2個を一緒にBへ運ぶことを評価条件として事前に定めた。赤は全工程の保護対象とした。システム構成を図1に示す。")
    add(data.model["instruction"],"small")
    story.append(architecture())
    add("図1　システムの処理構成。Qwenは操作対象と順序を含むJSON計画を生成する。低層制御は事前定義した操作とシミュレータの真値座標を用い，成否はモデル出力から分離した評価器で判定する。","caption")

    # Page 2: engineering methods, including the exact boundary of AI use.
    page();heading("3. 方法")
    heading("3.1 視覚言語モデルによる計画生成",2)
    add("事前学習済みのQwen3-VL-8B-Instruct [1] に，実際の初期カメラ画像，指示全文，物体・トレーの識別子と保護対象を与えた。操作語彙はpick_place，push_out，stack，move_stackと終了を表すdoneに限定した。正解の操作列はプロンプトへ与えず，モデルが選んだ対象・支持物・目的地を含むJSONを厳密に検査した。追加学習は行わず，do_sample=False，最大生成512トークンで1回だけ計画を生成した。")
    add("実際の出力は，青→A，黄→B，青の押し出し，黄を青へ積み重ね，積層した2個→B，doneの順であった。元のモデル回答，検査後の計画，実行した引数を照合した。赤への操作は，モデルへの指示に加えてプログラム側でも拒否する。")
    heading("3.2 逆運動学・経路計画・把持制御",2)
    add("Genesis [2,3] の逆運動学（inverse kinematics）に手先の目標位置・姿勢を与え，関節角度を求めた。空中の移動には衝突検査付きの経路計画を用い，接触を伴う下降・押し出し・積層搬送には小さな直交座標の移動を逐次求解した。求めた関節目標を位置制御へ渡し，指は緩やかに閉じた後，通常2 N，2個同時把持では4 Nの力を指ごとに与えた。操作対象の座標はシミュレータから読み取り，画像から座標を推定する処理は実装していない。")
    add("Pandaモデル [4] には，幅50 mm，厚さ6 mm，長さ60 mm，指当たり追加質量19.8 gの衝突形状を持つ延長指を追加した。標準のPanda指とは異なる構成であり，派生モデル内の質量・慣性も更新した。2個搬送では両方の物体を指で挟む。物体を指へ固定する溶接拘束や実行中の座標の強制代入は用いず，接触と重力により運動させた。")
    heading("3.3 位置フィードバックと成否判定",2)
    add("積み重ね直前には，上側物体と支持物体の位置誤差を測り，最大4回，補正移動の総量30 mm以内で手先位置を調整した。条件を満たせない場合は停止する。これは事前に実装した位置フィードバックであり，実行中にモデルがコードを修正したり，各操作後に計画を再生成したりする構成ではない。")
    add("青と黄の中心座標をp_b(t)=(x_b,y_b,z_b)，p_y(t)=(x_y,y_y,z_y)とする。積層の水平ずれをd_xy(t)=√{(x_y−x_b)²+(y_y−y_b)²}，中心高さ差をΔz(t)=z_y−z_bと定義した。搬送中はd_xy≤8 mm，|Δz−25 mm|≤6 mmを全記録点で要求する。赤の変位は初期位置からのユークリッド距離の最大値で評価した。")
    add("表1　段階評価の主要条件。記録点の位置と実行した操作時刻から判定する。","caption")
    table([
        ["判定対象","事前に定めた条件"],
        ["配置・終端の安定性","目標内に配置し，0.5秒以上，位置変動5 mm以下"],
        ["押し出し","水平移動40 mm以上，中心上昇30 mm以下，物体全体がAの外"],
        ["搬送中・終端の積層","水平ずれ8 mm以下，高さ差25±6 mm"],
        ["保護・記録間隔","赤の最大変位10 mm以下，記録間隔0.20秒以下"],
    ],[122,W-122])

    # Page 3: real observations, all from the one completed full run.
    page();heading("4. 実験条件と動作過程")
    add("設定と評価条件を固定した後，2026年10月29日に1回の正式試行を実行した。推論と描画にはNVIDIA RTX PRO 6000 Blackwell Max-Qを用い，物理演算はCPUで行った。ソフトウェアと数値設定を表2に，実行前後の場面を図2に示す。")
    add("表2　正式試行の実行条件。モデル・ライブラリの版は実行時記録に基づく。","caption")
    table([
        ["項目","設定"],
        ["モデル・推論","Qwen3-VL-8B-Instruct / bfloat16 / max_pixels=65,536"],
        ["実行環境","Python 3.11.15 / PyTorch 2.8.0+cu128 / Transformers 5.12.0"],
        ["シミュレーション","Genesis 1.4.4 / 時間刻み0.01秒 / サブステップ4 / no-slip反復5"],
        ["観測・評価","画像960×720画素 / 位置履歴1,202点 / 目標記録周期0.1秒"],
    ],[98,W-98])
    panel_width=(W-12)/2
    def montage(selected, *, crop=False, labels=None):
        grid=[]
        for n in range(0,len(selected),2):
            row=[]
            for j,frame in enumerate(selected[n:n+2]):
                image_path=figures/frame["image"]
                ratio=.75
                if crop:
                    image_path=figures/(Path(frame["image"]).stem+"-detail.png")
                    with PILImage.open(figures/frame["image"]) as im:
                        im.crop((300,210,730,660)).save(image_path)
                    ratio=450/430
                label=labels[n+j] if labels else frame["caption"]
                row.append([Image(str(image_path),width=panel_width,height=panel_width*ratio),
                            para(f"{label}　t={frame['task_elapsed_s']:.1f} s","small",record=False)])
            grid.append(row)
        block=Table(grid,colWidths=[panel_width+6,panel_width+6],hAlign="LEFT")
        block.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),0),
                                  ("RIGHTPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),4)]))
        story.append(block)
    montage([frames[0],frames[5]],labels=["(a) 初期状態","(b) 最終状態"])
    add("図2　同一の正式試行における実行前後の実際のカメラ画像。原画像をグレースケール化し，保存された物体座標を固定カメラへ投影して記号を重ねた。bは青，yは黄，rは赤，A・Bはトレーを表す。時刻tは初期安定化後の課題開始を0秒とする。最終状態では，青を下，黄を上にした2個がB内にある。","caption")
    heading("4.1 記録と評価の対応",2)
    add(f"課題開始から終了までの仮想時間は{data.timing['simulation']:.2f}秒であった。各操作の開始・終了時刻，元のモデル出力，実行した引数，位置履歴，カメラ画像を同一試行に対応付けて保存した。位置記録は{len(trajectory):,}点で，実測した最大記録間隔は{max_sample_gap:.2f}秒であった。提出用の編集動画とは別に，121秒の原動画と評価に用いた記録も公開資料に含めた。")

    page();heading("4.2 多段階操作の実行過程",2)
    add("初期分類から2個同時把持までの過程を図3に示す。画像は原動画の該当時刻から抽出し，作業領域を切り出した。押し出し後の青を支持物体として黄を解放し，積層の安定を確認してから両方を挟んで持ち上げた。最終的な解放後の状態は図2(b)に示した。")
    montage(frames[1:5],crop=True,labels=["(a) 青をA，黄をBへ配置","(b) 青をAの外へ押し出す","(c) 黄を青の上へ積む","(d) 2個を一緒に持ち上げる"])
    add("図3　分類，押し出し，積み重ね，同時把持の実際の動作過程。図2と同じ試行・記号を用い，4枚とも元画像の同一領域を同じ倍率で切り出した。各図の時刻は動作中または各段階の終端付近である。図3(d)の後，2個を同時に運び，Bへ配置してから指を開く。","caption")

    # Page 4: reproducible numerical results and actual recorded trajectories.
    page();heading("5. 結果")
    add("5操作はすべて完了し，初期配置2項目，分類完了，押し出し，積み重ね，搬送中の積層維持，最終配置の7項目で条件を満たした。判定はモデルのdone出力ではなく，保存した位置履歴と操作時刻から再計算した。主要な測定値を表3，位置履歴を図4に示す。")
    add("表3　正式試行の測定結果。位置に関する数値はシミュレータ内の記録値である。","caption")
    table([
        ["評価量","記録値"],
        ["段階評価 / 正式試行数","7項目達成 / 1回"],
        ["青の押し出し距離 / 最大中心上昇",f"{milestones['push_blue_outside_a']['planar_displacement_m']*1000:.3f} mm / {milestones['push_blue_outside_a']['maximum_center_rise_m']*1000:.3f} mm"],
        ["2個の持ち上げ量（青 / 黄）",f"{lift['blue_block']*1000:.3f} mm / {lift['yellow_block']*1000:.3f} mm"],
        ["搬送中の最大水平ずれ / 記録数",f"{max(offsets):.3f} mm / {len(carried)}点"],
        ["搬送中の中心高さ差",f"{min(gaps):.3f}～{max(gaps):.3f} mm"],
        ["最終積層の水平ずれ / 中心高さ差",f"{final_xy:.3f} mm / {final_gap:.3f} mm"],
        ["赤の最大変位",f"{red_m:.3e} m（数値誤差程度）"],
        ["仮想時間 / 全工程の実時間",f"{data.timing['simulation']:.2f}秒 / {data.timing['wall']:.2f}秒"],
        ["VLM工程の待ち時間 / 生成時間",f"{data.timing['model_wait']:.2f}秒 / {data.timing['generation']:.2f}秒"],
    ],[227,W-227])
    story.append(result_plot(trajectory,transport["started_s"],transport["ended_s"]))
    add("図4　記録した中心座標の時系列。(a)は青を実線，黄を破線で示す。点線は各操作の終了時刻である。(b)(c)は最終搬送区間の389点を拡大し，記録点を線で結んだ。水平ずれの許容上限は8 mm，高さ差の許容範囲は19～31 mmであり，拡大図の縦軸範囲とは異なる。記録点間の動作や実機での測定精度を保証する図ではない。","caption")

    # Page 5: interpretation, limitations, reproduction and explicit sources.
    page();heading("6. 考察")
    add("本事例では，一つの指示から生成した計画が，分類後に同じ物体を再操作する5段階の物理動作につながった。工夫した点は，支持物体を明示した操作語彙，保護対象の検査，実際の計画と記録の対応付け，積み重ね前の位置補正，2物体を同時に挟める延長指である。表3の値は，設定した条件下で積層を保って搬送できたことを支持する。")
    add("設定確定前には，8回の制御校正で把持・積層・搬送を調整した。このほか，旧動作スキーマとの不一致1件と，資源確認段階での実行前拒否1件を開発上の失敗として別に記録した。これら10件は正式試行の分母に混ぜない。一方，調整後に選んだ1回だけから一般的な成功率を推定することもできない。")
    heading("6.1 適用範囲と制約",2)
    add("評価対象は固定配置の3物体と一つの混合言語指示に限られる。各言語での反復試験，配置変更，未知物体，遮蔽，外乱，実機での検証は行っていない。画像なし条件などの比較実験もなく，視覚入力の寄与は定量化していない。低層制御は真値座標と事前定義した操作に依存し，Qwenは初期画像から計画を1回生成する。したがって本システムは，画像から連続関節動作を直接出力する学習済みの視覚言語行動モデルではない。")
    add("赤の変位は数値誤差程度であったが，これはシミュレータの記録値であり，実機のナノメートル精度を意味しない。積層は記録点ごとに判定したため，接触力の分布や未観測の瞬間を位置履歴だけでは立証できない。今後は，配置を変えた反復試験，画像からの位置推定，各操作後の再観測と再計画を追加して適用範囲を評価する必要がある。")
    heading("7. 結論")
    add("Qwen3-VLによる混合言語指示の計画生成とGenesisによる接触動作を接続し，赤を保護しながら分類，押し出し，積み重ね，2物体の同時搬送を行う課題を実装した。固定条件での1回の正式試行は7項目の評価条件を満たした。結論はこの実装と事例の範囲に限定する。")
    heading("再現用資料")
    add("公開リポジトリ [5] にコード，依存関係，課題・評価条件，モデル出力，実行した計画，位置履歴と原動画を収録した。モデル重みは含めず，公式配布元の同一revisionを利用する。提出動画は70.50秒で，動作部分を2倍速と表示し，全操作を同一試行から収録した。ソース識別と詳細な帰属はリポジトリ内の記録を参照されたい。","small")
    heading("参考文献（閲覧日：2026年10月29日）")
    references=[
        ("[1] Qwen Team. Qwen3-VL-8B-Instruct, model card. Revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b.",
         "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/tree/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
        ("[2] Genesis-Embodied-AI. Genesis World, version 1.4.4. Software repository.","https://github.com/Genesis-Embodied-AI/genesis-world"),
        ("[3] Genesis World. Inverse kinematics and motion planning. Official documentation, v1.4.0.","https://genesis-world.readthedocs.io/en/v1.4.0/user_guide/robot_control/inverse_kinematics_motion_planning.html"),
        ("[4] Franka Robotics. franka_description. franka_ros, develop branch.","https://github.com/frankarobotics/franka_ros/tree/develop/franka_description"),
        ("[5] shigin123. 尝试VLA: Physical AI course project. Runtime evidence: submission-20261029.",REPO),
    ]
    for description,url in references:
        add(description,"small")
        all_text.append(url);text_log.append(url)
        # All links deliberately remain black in the paper.
        story.append(Paragraph(f'<link href="{escape(url)}" color="black">{escape(url)}</link>',styles["small"]))

    text="\n\n".join(text_log)+"\n"
    require(not re.search(r"(?:/(?:Users|home|nas)/|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|73801|shigin26)",text),"Private text in academic report")
    require("codex" not in text.lower(),"Do not add a development-tool disclosure to this report")
    font_glyphs=set(pdfmetrics.getFont(FONT).face.charToGlyph) | {ord("・")}
    missing=sorted({c for paragraph in all_text for c in paragraph if not c.isspace() and ord(c) not in font_glyphs})
    require(not missing,"Missing report-font characters: "+"".join(missing))
    output.parent.mkdir(parents=True,exist_ok=True)
    manuscript.parent.mkdir(parents=True,exist_ok=True)
    manuscript.write_text(text,encoding="utf-8")
    def decorate(canvas,doc):
        canvas.saveState();canvas.setFillColor(BLACK);canvas.setStrokeColor(BLACK)
        canvas.setFont(FONT,8)
        canvas.drawString(24*72/25.4,A4[1]-14*72/25.4,"Physical AI 応用1講座 最終課題")
        canvas.drawRightString(A4[0]-24*72/25.4,A4[1]-14*72/25.4,account)
        canvas.setLineWidth(.35);canvas.line(24*72/25.4,16*72/25.4,A4[0]-24*72/25.4,16*72/25.4)
        canvas.setFont(LATIN,9);canvas.drawCentredString(A4[0]/2,11*72/25.4,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=24*72/25.4,leftMargin=24*72/25.4,
                          topMargin=20*72/25.4,bottomMargin=21*72/25.4,
                          title="混合言語指示による多段階ブロック操作",author=account,
                          subject="Physical AI 応用1 最終課題",creator="Academic report builder")
    doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
    for frame in frames:
        detail=figures/(Path(frame["image"]).stem+"-detail.png")
        if detail.is_file():
            frame["detail"]={"image":detail.name,"crop_box_original_pixels":[300,210,730,660],
                             "sha256":hashlib.sha256(detail.read_bytes()).hexdigest()}
    (figures/"figure-sources.json").write_text(json.dumps(frames,ensure_ascii=False,indent=2)+"\n")
    from pypdf import PdfReader
    pages=len(PdfReader(str(output)).pages)
    return {"pages":pages,"sha256":hashlib.sha256(output.read_bytes()).hexdigest(),
            "source_manifest_sha256":data.manifest_digest,"figure_count":4,"table_count":3,
            "glyph_check_passed":True,"private_text_check_passed":True,
            "metrics":{"max_transport_xy_mm":max(offsets),"gap_range_mm":[min(gaps),max(gaps)],
                       "final_xy_mm":final_xy,"final_gap_mm":final_gap,"red_displacement_m":red_m,
                       "actual_max_sample_gap_s":max_sample_gap},"visual_review_required":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,default=ROOT/"examples/validated_multistep_run")
    parser.add_argument("--output",type=Path,default=ROOT/"交付/論文形式レポート/説明資料_PAI最終課題_shigin123.pdf")
    parser.add_argument("--account",default="shigin123")
    parser.add_argument("--video",type=Path,default=ROOT/"交付/最终提交/PAI最終課題_shigin123.mp4")
    parser.add_argument("--font",type=Path)
    parser.add_argument("--font-index",type=int)
    args=parser.parse_args()
    require(re.fullmatch(r"[A-Za-z0-9_.-]{1,64}",args.account) is not None,"Invalid account identifier")
    require(args.output.name==f"説明資料_PAI最終課題_{args.account}.pdf","Use the exact course report filename")
    register_fonts(args.font,args.font_index)
    data=load_evidence(args.run_dir,args.run_dir/"trial-summary.json",args.video)
    qa_dir=ROOT/"reports/academic-report-review"
    receipt=build(data,args.output,args.account,qa_dir/"figures",ROOT/"docs/ACADEMIC_REPORT_JA.md")
    (qa_dir/"build-receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
