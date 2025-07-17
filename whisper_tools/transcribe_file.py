#!/usr/bin/env python3
"""
统一的 Whisper 转写模块，支持音频和视频文件的自动识别与转写。

该模块提供了一个标准接口 transcribe_file()，可用于：
- 小程序用户上传音频后自动识别
- 训练系统对讲解视频批量转写
- 智能体调用进行语音理解与字幕生成
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import whisper
from moviepy.editor import VideoFileClip

# ---------------------------------------------------------------------------
# 🔧 环境配置（继承自原 transcriber.py）
# ---------------------------------------------------------------------------
FFMPEG_PATH = "/usr/local/bin/ffmpeg"  # 可通过环境变量覆盖

# 确保 FFmpeg 可用
if os.path.isfile(FFMPEG_PATH):
    ffmpeg_dir = os.path.dirname(FFMPEG_PATH)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    os.environ["FFMPEG_BINARY"] = FFMPEG_PATH

# 自愈 mel_filters.npz 缺失问题
ASSETS_DST = Path(__file__).resolve().parent.parent / "whisper" / "assets" / "mel_filters.npz"
ASSETS_SRC = Path.home() / ".cache" / "whisper" / "assets" / "mel_filters.npz"

if not ASSETS_DST.exists() and ASSETS_SRC.exists():
    ASSETS_DST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS_SRC, ASSETS_DST)

# ---------------------------------------------------------------------------
# 🎵 文件类型识别
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

def is_audio(file_path: str) -> bool:
    """判断文件是否为音频格式"""
    return Path(file_path).suffix.lower() in AUDIO_EXTENSIONS

def is_video(file_path: str) -> bool:
    """判断文件是否为视频格式"""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS

# ---------------------------------------------------------------------------
# 🎤 Whisper 模型管理
# ---------------------------------------------------------------------------
# 全局模型实例，避免重复加载
_model = None

def get_whisper_model(model_name: str = "base") -> whisper.Whisper:
    """获取或加载 Whisper 模型（单例模式）"""
    global _model
    if _model is None:
        _model = whisper.load_model(model_name)
    return _model

# ---------------------------------------------------------------------------
# 🎬 核心转写函数
# ---------------------------------------------------------------------------
def transcribe_file(
    file_path: str, 
    language: Optional[str] = None,
    model_name: str = "base"
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    自动识别音频或视频类型，提取音频并使用 Whisper 模型转写。
    
    参数:
        file_path: 输入的音频或视频路径（支持 .mp3, .wav, .mp4, .mov 等）
        language: 可选，指定语种；若为 None，自动识别语言
        model_name: Whisper 模型名称，默认为 "base"
    
    返回:
        text: 转写的完整文本
        segments: 每段文字及其时间戳，格式为 [{"start": 0.0, "end": 2.5, "text": "..."}, ...]
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
        Exception: 转写过程中的其他错误
    """
    # 检查文件是否存在
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 获取模型
    model = get_whisper_model(model_name)
    
    # 根据文件类型处理
    if is_video(str(file_path)):
        # 视频文件：先提取音频
        result = _transcribe_video(str(file_path), model, language)
    elif is_audio(str(file_path)):
        # 音频文件：直接转写
        result = _transcribe_audio(str(file_path), model, language)
    else:
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")
    
    # 提取文本和分段信息
    text = result["text"]
    segments = result["segments"]
    
    return text, segments

def _transcribe_audio(
    file_path: str, 
    model: whisper.Whisper, 
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    对音频文件进行 Whisper 转写
    
    返回：result 包含 text 和 segments
    """
    kwargs = {}
    if language:
        kwargs["language"] = language
    
    result = model.transcribe(file_path, **kwargs)
    return result

def _transcribe_video(
    video_path: str, 
    model: whisper.Whisper, 
    language: Optional[str] = None
) -> Dict[str, Any]:
    """
    从视频中提取音频并进行 Whisper 转写
    """
    temp_audio = None
    try:
        # 创建临时音频文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            temp_audio = tmp_audio.name
            
            # 提取音频
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio)
            video.close()  # 释放视频资源
            
            # 转写音频
            result = _transcribe_audio(temp_audio, model, language)
            
            return result
    finally:
        # 确保清理临时文件
        if temp_audio and os.path.exists(temp_audio):
            os.remove(temp_audio)

# ---------------------------------------------------------------------------
# 🔧 辅助函数（可选功能）
# ---------------------------------------------------------------------------
def save_as_txt(text: str, output_path: str) -> None:
    """保存转写文本为 .txt 文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_as_srt(segments: List[Dict[str, Any]], output_path: str) -> None:
    """保存转写结果为 .srt 字幕文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_time = _format_timestamp(segment["start"])
            end_time = _format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def _format_timestamp(seconds: float) -> str:
    """将秒数转换为 SRT 时间戳格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def save_as_json(result: Dict[str, Any], output_path: str) -> None:
    """保存完整的转写结果为 JSON 文件"""
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ---------------------------------------------------------------------------
# 🚀 CLI 接口（可选）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 Whisper 转写音频或视频文件")
    parser.add_argument("file", help="输入文件路径")
    parser.add_argument("--lang", default=None, help="指定语言代码（如 zh, en）")
    parser.add_argument("--model", default="base", help="Whisper 模型名称")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--formats", nargs="+", default=["txt"], 
                        choices=["txt", "srt", "json"], help="输出格式")
    
    args = parser.parse_args()
    
    try:
        # 执行转写
        print(f"🎤 正在转写: {args.file}")
        text, segments = transcribe_file(args.file, language=args.lang, model_name=args.model)
        
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # 保存结果
        base_name = Path(args.file).stem
        
        if "txt" in args.formats:
            txt_path = output_dir / f"{base_name}.txt"
            save_as_txt(text, str(txt_path))
            print(f"✅ 文本已保存: {txt_path}")
        
        if "srt" in args.formats:
            srt_path = output_dir / f"{base_name}.srt"
            save_as_srt(segments, str(srt_path))
            print(f"✅ 字幕已保存: {srt_path}")
        
        if "json" in args.formats:
            json_path = output_dir / f"{base_name}.json"
            save_as_json({"text": text, "segments": segments}, str(json_path))
            print(f"✅ JSON 已保存: {json_path}")
            
    except Exception as e:
        print(f"❌ 转写失败: {e}")
        exit(1)