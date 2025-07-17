# translator.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载 GOOGLE_TRANSLATE_API_KEY

GOOGLE_API_KEY = os.getenv("GOOGLE_TRANSLATE_API_KEY")
TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

def translate_text(text, source_lang="en", target_lang="zh"):
    if not GOOGLE_API_KEY:
        raise ValueError("未检测到 GOOGLE_TRANSLATE_API_KEY，请检查 .env 文件")

    payload = {
        'q': text,
        'source': source_lang,
        'target': target_lang,
        'format': 'text',
        'key': GOOGLE_API_KEY
    }

    response = requests.post(TRANSLATE_URL, data=payload)
    if response.status_code == 200:
        data = response.json()
        return data["data"]["translations"][0]["translatedText"]
    else:
        raise Exception(f"翻译失败：{response.status_code} {response.text}")

# ✅ 命令行调试支持
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="翻译文本")
    parser.add_argument("text", help="需要翻译的文本")
    parser.add_argument("--lang", default="zh", help="目标语言（默认 zh）")
    args = parser.parse_args()

    translated = translate_text(args.text, "en", args.lang)
    print("=== 翻译结果 ===\n")
    print(translated)
