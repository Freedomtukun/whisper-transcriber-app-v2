# scripts/dual_text.py
import argparse

def merge_texts(en_file, zh_file, output_file):
    with open(en_file, "r", encoding="utf-8") as f1:
        en_lines = f1.readlines()
    with open(zh_file, "r", encoding="utf-8") as f2:
        zh_lines = f2.readlines()

    merged = []
    for en, zh in zip(en_lines, zh_lines):
        merged.append(f"{en.strip()} || {zh.strip()}")

    with open(output_file, "w", encoding="utf-8") as out:
        out.write("\n".join(merged))
    print(f"✅ 中英对照文件已保存：{output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成中英对照文本")
    parser.add_argument("en_file", help="英文原文路径")
    parser.add_argument("zh_file", help="中文翻译路径")
    parser.add_argument("output_file", help="输出路径")
    args = parser.parse_args()

    merge_texts(args.en_file, args.zh_file, args.output_file)
