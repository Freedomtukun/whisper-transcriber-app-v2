"""
Whisper Tools - 音频和视频转写工具集

这个包提供了基于 OpenAI Whisper 模型的转写功能：
- transcribe_file: 单文件转写功能
- transcribe_batch: 批量文件转写功能
"""

__version__ = "1.0.0"
__author__ = "Whisper Tools Team"

# 导入主要功能
from .transcribe_file import (
    transcribe_file,
    save_as_txt,
    save_as_srt,
    save_as_json,
    is_audio,
    is_video
)

# 定义公开的 API
__all__ = [
    "transcribe_file",
    "save_as_txt",
    "save_as_srt",
    "save_as_json",
    "is_audio",
    "is_video"
]