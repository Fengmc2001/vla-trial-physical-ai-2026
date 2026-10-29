# 尝试VLA｜Physical AI 最終課題

Qwen3-VLが画像と指示から操作を計画し，Genesis内のロボットアームがブロックを動かす課題です。

## 動画・説明資料

- [デモ動画（YouTube・70.5秒）](https://www.youtube.com/watch?v=hS34eNkm5VM)
- [説明資料（PDF・4ページ）](delivery/説明資料_PAI最終課題_shigin123.pdf)
- [説明資料のLaTeX原稿](report/)

## 実行したタスク

青をAへ，黄をBへ配置した後，青をAの外へ押し出し，黄を青の上に積み，2個を一緒にBへ運びます。赤いブロックは全工程で保護します。

追加で，青を持ち上げて置く操作を3回繰り返す例も記録しました。動画は5段階タスク，説明資料は両方の例を扱っています。

## AIと制御の役割

画像・指示 → **QwenのJSON計画** → 対象・引数の検査 → **制御プログラムとGenesis**

Qwenは操作の種類・対象・順序を選びます。制御側はシミュレータの現在位置を読み直し，手先と関節の運動を計算します。追加学習は行っていません。

固定した場面での実行例です。5段階タスクは7項目の確認を通過し，追加の反復操作は3回の実行完了を確認しました。

## コード・使い方

- [ソースコード](src/vla_trial/) ／ [タスク設定](configs/tasks/) ／ [実行記録](examples/)
- [環境構築・実行手順](docs/TECHNICAL_REFERENCE.md) ／ [中文使用说明](docs/USER_GUIDE_ZH.md)
- [利用したモデル・ソフトウェアと出典](THIRD_PARTY_NOTICES.md)

<!-- Handoff: 2026-10-29 · Codex · Verified the current four-page PDF; simplified README and moved detailed setup to docs/TECHNICAL_REFERENCE.md. -->
