"""CPU-only preparation UI. Intentionally exposes no runtime-start endpoint."""
import importlib.abc
import os
import sys


class PreparationImportGuard(importlib.abc.MetaPathFinder):
    blocked = frozenset(("torch", "torchvision", "transformers", "genesis", "quadrants", "taichi", "cupy", "pycuda", "tensorflow", "jax"))

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ImportError(f"PREPARATION_ONLY: blocked import {fullname}")
        return None


def build_ui(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if not any(isinstance(x, PreparationImportGuard) for x in sys.meta_path):
        sys.meta_path.insert(0, PreparationImportGuard())
    import gradio as gr
    from .diagnostics import diagnose
    from .preview import FIXTURES, fixture_plan, schematic
    with gr.Blocks(title="尝试VLA · Environment preparation", analytics_enabled=False) as demo:
        gr.Markdown("# 尝试VLA · 環境準備\n**準備モード：GPU・VLM推論・物理シミュレーションは実行しません。**\n\n这是环境和接口预览；下方示意图与手写样例不能作为最终课题的机器人执行证据。")
        with gr.Row():
            gr.Image(value=schematic(), label="静态布局示意 / Static configuration illustration", interactive=False, type="pil")
            with gr.Column():
                language = gr.Dropdown(list(FIXTURES), value="日本語", label="样例语言")
                instruction = gr.Textbox(value=FIXTURES["日本語"], lines=4, label="任务指令草稿（本页面不送入模型）")
                plan = gr.JSON(value=fixture_plan("日本語"), label="手写接口样例 / NOT model output")
                language.change(lambda lang: (FIXTURES[lang], fixture_plan(lang)), language, [instruction, plan], api_name=False)
        with gr.Accordion("依赖与模型缓存元数据（不加载权重）", open=False):
            report = gr.JSON(value=diagnose(config), label="Metadata-only status")
            refresh = gr.Button("刷新元数据检查 / Refresh metadata")
            refresh.click(lambda: diagnose(config), outputs=report, api_name=False)
        gr.Markdown("### 下一步\n1. 完成指令与场景设计。\n2. 另行确认 GPU 授权、准入和渲染设备。\n3. 验证实际抓放、视觉反馈与录制。\n4. 真实测试后填写报告、整理 public 仓库及 1–3 分钟视频。\n\n当前没有启动模型或仿真的按钮，也不自动上传、发布或生成公开分享链接。")
    return demo
