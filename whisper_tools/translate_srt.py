# scripts/translate_srt.py
import argparse
from srt_utils import load_srt, translate_srt, save_srt
from translator import translate_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="翻译 .srt 字幕文件")
    parser.add_argument("input", help="输入 SRT 路径")
    parser.add_argument("output", help="输出 SRT 路径")
    parser.add_argument("--lang", default="zh", help="目标语言（默认：zh）")
    args = parser.parse_args()

    subs = load_srt(args.input)
    translated = translate_srt(subs, lambda x: translate_text(x, "en", args.lang))
    save_srt(translated, args.output)
    print(f"✅ 翻译完成：{args.output}")
