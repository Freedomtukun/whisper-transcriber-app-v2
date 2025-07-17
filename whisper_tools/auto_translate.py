# scripts/auto_translate.py
import argparse
import srt
import datetime
from transcriber import transcribe_file
from translator import translate_text
from srt_utils import save_srt

def segments_to_srt(segments):
    subtitles = []
    for i, seg in enumerate(segments):
        start = datetime.timedelta(seconds=seg['start'])
        end = datetime.timedelta(seconds=seg['end'])
        subtitles.append(srt.Subtitle(index=i+1, start=start, end=end, content=seg['text']))
    return subtitles

def auto_translate(file_path, target_lang="zh"):
    text, segments = transcribe_file(file_path)
    srt_entries = segments_to_srt(segments)
    translated = [translate_text(entry.content, "en", target_lang) for entry in srt_entries]
    for i, entry in enumerate(srt_entries):
        entry.content = translated[i]
    output_path = f"output/{file_path.split('/')[-1].split('.')[0]}_{target_lang}.srt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(srt_entries))
    print(f"✅ 翻译完成，已保存至：{output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动识别+翻译字幕")
    parser.add_argument("file", help="音频或视频文件路径")
    parser.add_argument("--lang", default="zh", help="目标语言（默认：zh）")
    args = parser.parse_args()
    auto_translate(args.file, args.lang)
