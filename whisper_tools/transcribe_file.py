#!/usr/bin/env python3
"""
统一的 Whisper 转写模块，支持音频和视频文件的自动识别与转写。

该模块提供了一个标准接口 transcribe_file()，可用于：
- 小程序用户上传音频后自动识别
- 训练系统对讲解视频批量转写
- 智能体调用进行语音理解与字幕生成

依赖安装：
pip install opencc-python-reimplemented colorama
"""

import os
import shutil
import tempfile
import time
import logging
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from functools import lru_cache
import whisper
from moviepy.editor import VideoFileClip
from opencc import OpenCC

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False
    Fore = Style = type('', (), {'GREEN': '', 'YELLOW': '', 'RED': '', 'CYAN': '', 'RESET_ALL': ''})()

# ---------------------------------------------------------------------------
# 🔧 环境配置
# ---------------------------------------------------------------------------
FFMPEG_PATH = "/usr/local/bin/ffmpeg"

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
# 📊 数据结构定义
# ---------------------------------------------------------------------------
@dataclass
class TranscriptionMetadata:
    """转写元数据结构"""
    detected_language: str
    model_name: str
    file_type: str
    file_size_mb: float
    duration_seconds: float
    processing_time_seconds: float
    keep_traditional: bool
    segments_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)

# ---------------------------------------------------------------------------
# 🎵 文件类型识别
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

def is_audio(file_path: Union[str, Path]) -> bool:
    """判断文件是否为音频格式"""
    return Path(file_path).suffix.lower() in AUDIO_EXTENSIONS

def is_video(file_path: Union[str, Path]) -> bool:
    """判断文件是否为视频格式"""
    return Path(file_path).suffix.lower() in VIDEO_EXTENSIONS

def is_supported_format(file_path: Union[str, Path]) -> bool:
    """判断文件是否为支持的音视频格式"""
    return is_audio(file_path) or is_video(file_path)

# ---------------------------------------------------------------------------
# 🎤 模型与转换器管理（懒加载对象池）
# ---------------------------------------------------------------------------
class ModelPool:
    """Whisper 模型对象池，支持懒加载和缓存"""
    _models = {}
    
    @classmethod
    def get_model(cls, model_name: str = "base") -> whisper.Whisper:
        """获取或加载 Whisper 模型"""
        if model_name not in cls._models:
            try:
                logging.info(f"正在加载 Whisper 模型: {model_name}")
                cls._models[model_name] = whisper.load_model(model_name)
                logging.info(f"模型加载成功: {model_name}")
            except Exception as e:
                raise RuntimeError(f"Whisper 模型加载失败 ({model_name}): {e}")
        return cls._models[model_name]

@lru_cache(maxsize=1)
def get_opencc_converter() -> OpenCC:
    """获取 OpenCC 转换器实例（LRU 缓存）"""
    try:
        return OpenCC('t2s')
    except Exception as e:
        raise RuntimeError(f"OpenCC 初始化失败: {e}")

# ---------------------------------------------------------------------------
# 📝 繁简转换辅助函数（建议迁移至 zh_utils.py）
# ---------------------------------------------------------------------------
def convert_segments_to_simplified(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """将 segments 中的繁体中文转换为简体中文"""
    if not segments:
        return segments
    
    cc = get_opencc_converter()
    return [
        {**segment, "text": cc.convert(segment["text"])}
        for segment in segments
    ]

def convert_text_to_simplified(text: str) -> str:
    """将文本从繁体中文转换为简体中文"""
    if not text.strip():
        return text
    
    cc = get_opencc_converter()
    return cc.convert(text)

# ---------------------------------------------------------------------------
# 🔧 上下文管理器
# ---------------------------------------------------------------------------
@contextmanager
def extract_audio_from_video(video_path: Union[str, Path]):
    """从视频文件提取音频的上下文管理器"""
    temp_audio = None
    video_clip = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            temp_audio = tmp_audio.name
            
        logging.info("正在从视频提取音频...")
        video_clip = VideoFileClip(str(video_path))
        
        if video_clip.audio is None:
            raise RuntimeError("视频文件中未找到音频轨道")
            
        video_clip.audio.write_audiofile(
            temp_audio, 
            verbose=False, 
            logger=None,
        )
        
        yield temp_audio
        
    except Exception as e:
        raise RuntimeError(f"音频提取失败: {e}")
    finally:
        if video_clip:
            try:
                video_clip.close()
            except:
                pass
        
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except:
                pass

# ---------------------------------------------------------------------------
# 🎬 核心转写函数
# ---------------------------------------------------------------------------
def transcribe_file(
    file_path: Union[str, Path], 
    language: Optional[str] = None,
    model_name: str = "base",
    keep_traditional: bool = False,
    verbose: bool = True
) -> Tuple[str, List[Dict[str, Any]], TranscriptionMetadata]:
    """
    自动识别音频或视频类型，提取音频并使用 Whisper 模型转写。
    
    参数:
        file_path: 输入的音频或视频路径
        language: 可选，指定语种；若为 None，自动识别语言
        model_name: Whisper 模型名称，默认为 "base"
        keep_traditional: 是否保留繁体中文，默认 False（转为简体）
        verbose: 是否显示详细信息
    
    返回:
        text: 转写的完整文本
        segments: 每段文字及其时间戳
        metadata: 转写元数据对象
    
    异常:
        FileNotFoundError: 文件不存在
        ValueError: 不支持的文件格式
        RuntimeError: 转写过程中的错误
    """
    file_path_obj = Path(file_path).resolve()
    
    # 预检查
    if not file_path_obj.exists():
        raise FileNotFoundError(f"文件不存在: {file_path_obj}")
    
    if not is_supported_format(file_path_obj):
        raise ValueError(f"不支持的文件格式: {file_path_obj.suffix}")
    
    start_time = time.time()
    
    try:
        model = ModelPool.get_model(model_name)
        
        # 根据文件类型处理
        if is_video(file_path_obj):
            logging.info("检测到视频文件，正在提取音频...")
            result = _transcribe_video(file_path_obj, model, language)
        else:
            logging.info("检测到音频文件，开始转写...")
            result = _transcribe_audio(file_path_obj, model, language)
        
        elapsed_time = time.time() - start_time
        
        # 构建元数据
        metadata = TranscriptionMetadata(
            detected_language=result.get("language", "unknown"),
            model_name=model_name,
            file_type="video" if is_video(file_path_obj) else "audio",
            file_size_mb=round(file_path_obj.stat().st_size / (1024*1024), 2),
            duration_seconds=result.get("segments", [{}])[-1].get("end", 0) if result.get("segments") else 0,
            processing_time_seconds=round(elapsed_time, 2),
            keep_traditional=keep_traditional,
            segments_count=len(result.get("segments", []))
        )
        
        logging.info(f"检测语言: {metadata.detected_language}")
        logging.info(f"处理耗时: {metadata.processing_time_seconds}秒")
        
        # 繁简转换
        if keep_traditional:
            return result["text"], result["segments"], metadata
        else:
            logging.info("正在转换为简体中文...")
            simplified_text = convert_text_to_simplified(result["text"])
            simplified_segments = convert_segments_to_simplified(result["segments"])
            return simplified_text, simplified_segments, metadata
            
    except Exception as e:
        raise RuntimeError(f"转写过程失败: {e}")

def _transcribe_audio(
    file_path: Path, 
    model: whisper.Whisper, 
    language: Optional[str] = None
) -> Dict[str, Any]:
    """对音频文件进行 Whisper 转写"""
    kwargs = {}
    if language:
        kwargs["language"] = language
        logging.info(f"使用指定语言: {language}")
    
    try:
        result = model.transcribe(str(file_path), **kwargs)
        return result
    except Exception as e:
        raise RuntimeError(f"音频转写失败 - {type(e).__name__}: {e}")

def _transcribe_video(
    video_path: Path, 
    model: whisper.Whisper, 
    language: Optional[str] = None
) -> Dict[str, Any]:
    """从视频中提取音频并进行 Whisper 转写"""
    try:
        with extract_audio_from_video(video_path) as temp_audio:
            logging.info("音频提取完成，开始转写...")
            return _transcribe_audio(Path(temp_audio), model, language)
    except Exception as e:
        raise RuntimeError(f"视频转写失败 - {type(e).__name__}: {e}")

# ---------------------------------------------------------------------------
# 🔧 文件输出函数（建议迁移至 io_utils.py）
# ---------------------------------------------------------------------------
def save_as_txt(text: str, output_path: Union[str, Path]) -> None:
    """保存转写文本为 .txt 文件"""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, "w", encoding="utf-8") as f:
        f.write(text)

def save_as_srt(segments: List[Dict[str, Any]], output_path: Union[str, Path], max_line_length: int = 40) -> None:
    """保存转写结果为 .srt 字幕文件"""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, 1):
            start_time = _format_timestamp(segment["start"])
            end_time = _format_timestamp(segment["end"])
            text = segment["text"].strip()
            
            # 控制字幕行长度
            if len(text) > max_line_length:
                # 简单换行处理
                mid = len(text) // 2
                # 寻找最近的空格或标点进行换行
                for offset in range(10):
                    if mid + offset < len(text) and text[mid + offset] in ' ，。！？':
                        text = text[:mid + offset + 1] + '\n' + text[mid + offset + 1:]
                        break
                    elif mid - offset >= 0 and text[mid - offset] in ' ，。！？':
                        text = text[:mid - offset + 1] + '\n' + text[mid - offset + 1:]
                        break
            
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

def save_as_json(text: str, segments: List[Dict[str, Any]], metadata: TranscriptionMetadata, output_path: Union[str, Path]) -> None:
    """保存完整的转写结果为 JSON 文件"""
    import json
    
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "text": text,
        "segments": segments,
        "metadata": metadata.to_dict()
    }
    
    with open(output_path_obj, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

def save_language_info(metadata: TranscriptionMetadata, output_path: Union[str, Path]) -> None:
    """保存语言检测结果到单独文件"""
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path_obj, "w", encoding="utf-8") as f:
        f.write(f"检测语言: {metadata.detected_language}\n")
        f.write(f"模型: {metadata.model_name}\n")
        f.write(f"文件类型: {metadata.file_type}\n")
        f.write(f"处理耗时: {metadata.processing_time_seconds}秒\n")

# ---------------------------------------------------------------------------
# 🎨 美化输出函数
# ---------------------------------------------------------------------------
def print_colored(message: str, color: str = None) -> None:
    """彩色输出"""
    if HAS_COLORAMA and color:
        color_code = getattr(Fore, color.upper(), '')
        print(f"{color_code}{message}{Style.RESET_ALL}")
    else:
        print(message)

def print_summary(metadata: TranscriptionMetadata, file_path: str) -> None:
    """打印转写结果摘要"""
    print_colored("\n📝 转写完成:", "green")
    print(f"   📄 文件: {Path(file_path).name}")
    print(f"   🌍 语言: {metadata.detected_language}")
    print(f"   📊 文字长度: 预计 {int(metadata.duration_seconds * 5)} 字符")  # 估算
    print(f"   🎬 分段数量: {metadata.segments_count} 段")
    print(f"   ⏱️ 处理耗时: {metadata.processing_time_seconds}秒")
    print(f"   🎯 文字格式: {'原始格式' if metadata.keep_traditional else '简体中文'}")

# ---------------------------------------------------------------------------
# 📁 批量处理函数
# ---------------------------------------------------------------------------
def process_directory(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    **kwargs
) -> List[Tuple[str, bool]]:
    """
    批量处理目录中的音视频文件
    
    返回:
        处理结果列表：[(文件名, 是否成功), ...]
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    # 查找支持的文件
    supported_files = []
    for ext in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        supported_files.extend(input_dir.glob(f"*{ext}"))
    
    print_colored(f"🔍 找到 {len(supported_files)} 个支持的文件", "cyan")
    
    for file_path in supported_files:
        try:
            print_colored(f"\n🎤 正在处理: {file_path.name}", "yellow")
            
            text, segments, metadata = transcribe_file(
                file_path,
                verbose=False,
                **kwargs
            )
            
            # 保存文件
            base_name = file_path.stem
            
            save_as_txt(text, output_dir / f"{base_name}.txt")
            save_as_srt(segments, output_dir / f"{base_name}.srt")
            save_as_json(text, segments, metadata, output_dir / f"{base_name}.json")
            
            print_colored(f"✅ 完成: {file_path.name}", "green")
            results.append((file_path.name, True))
            
        except Exception as e:
            print_colored(f"❌ 失败: {file_path.name} - {e}", "red")
            results.append((file_path.name, False))
    
    return results

# ---------------------------------------------------------------------------
# 🚀 增强版 CLI 接口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    parser = argparse.ArgumentParser(
        description="使用 Whisper 转写音频或视频文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python transcribe_file.py video.mp4
  python transcribe_file.py audio.wav --lang zh --model small
  python transcribe_file.py video.mp4 --formats txt srt json --print-text
  python transcribe_file.py --input-dir ./videos --output-dir ./results
        """
    )
    
    # 输入参数
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("file", nargs="?", help="输入文件路径")
    group.add_argument("--input-dir", help="批量处理：输入目录路径")
    
    # 转写参数
    parser.add_argument("--lang", default=None, help="指定语言代码（如 zh, en）")
    parser.add_argument("--model", default="base", 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper 模型名称")
    
    # 输出参数
    parser.add_argument("--output-dir", default=None, help="输出目录（默认为输入文件同目录）")
    parser.add_argument("--formats", nargs="+", default=["txt"], 
                        choices=["txt", "srt", "json"], help="输出格式")
    parser.add_argument("--srt-max-line-length", type=int, default=40,
                        help="SRT 字幕最大行长度")
    
    # 功能参数
    parser.add_argument("--keep-traditional", action="store_true", 
                        help="保留繁体中文输出（默认转为简体）")
    parser.add_argument("--export-lang", action="store_true",
                        help="导出语言检测结果到单独文件")
    parser.add_argument("--print-text", action="store_true",
                        help="在命令行直接输出转写文本")
    parser.add_argument("--quiet", action="store_true",
                        help="静默模式，减少输出信息")
    
    args = parser.parse_args()
    
    # 配置日志级别
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    try:
        if args.input_dir:
            # 批量处理模式
            if args.output_dir is None:
                args.output_dir = Path(args.input_dir) / "transcription_output"
            
            results = process_directory(
                args.input_dir,
                args.output_dir,
                language=args.lang,
                model_name=args.model,
                keep_traditional=args.keep_traditional
            )
            
            # 统计结果
            successful = sum(1 for _, success in results if success)
            total = len(results)
            print_colored(f"\n🎯 批量处理完成: {successful}/{total} 文件成功", "green")
            
        else:
            # 单文件处理模式
            if not args.quiet:
                print_colored(f"🎤 正在转写: {args.file}", "cyan")
            
            text, segments, metadata = transcribe_file(
                args.file, 
                language=args.lang, 
                model_name=args.model,
                keep_traditional=args.keep_traditional,
                verbose=not args.quiet
            )
            
            # 确定输出目录
            if args.output_dir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = Path(args.file).parent / "output"
            
            output_dir.mkdir(exist_ok=True)
            base_name = Path(args.file).stem
            
            # 保存文件
            if "txt" in args.formats:
                txt_path = output_dir / f"{base_name}.txt"
                save_as_txt(text, txt_path)
                if not args.quiet:
                    print_colored(f"✅ 文本已保存: {txt_path}", "green")
            
            if "srt" in args.formats:
                srt_path = output_dir / f"{base_name}.srt"
                save_as_srt(segments, srt_path, args.srt_max_line_length)
                if not args.quiet:
                    print_colored(f"✅ 字幕已保存: {srt_path}", "green")
            
            if "json" in args.formats:
                json_path = output_dir / f"{base_name}.json"
                save_as_json(text, segments, metadata, json_path)
                if not args.quiet:
                    print_colored(f"✅ JSON 已保存: {json_path}", "green")
            
            # 导出语言信息
            if args.export_lang:
                lang_path = output_dir / f"{base_name}_language.txt"
                save_language_info(metadata, lang_path)
                if not args.quiet:
                    print_colored(f"✅ 语言信息已保存: {lang_path}", "green")
            
            # 显示摘要
            if not args.quiet:
                print_summary(metadata, args.file)
            
            # 直接输出文本
            if args.print_text:
                print_colored("\n📝 转写文本:", "cyan")
                print("-" * 50)
                print(text)
                print("-" * 50)
            
    except KeyboardInterrupt:
        print_colored("\n⏹️ 用户中断操作", "yellow")
        exit(1)
    except Exception as e:
        print_colored(f"❌ 处理失败: {e}", "red")
        logging.exception("详细错误信息:")
        exit(1)


# ---------------------------------------------------------------------------
# 📋 模块分离建议
# ---------------------------------------------------------------------------
"""
建议的项目结构：

whisper_tools/
├── __init__.py
├── transcribe_file.py          # 主转写模块（当前文件）
├── zh_utils.py                 # 中文处理工具
├── io_utils.py                 # 文件输入输出工具
├── models.py                   # 数据模型定义
└── tests/
    ├── __init__.py
    ├── test_transcribe_file.py
    ├── test_zh_utils.py
    └── test_io_utils.py

zh_utils.py 内容：
- convert_text_to_simplified()
- convert_segments_to_simplified()
- get_opencc_converter()
- 其他中文处理函数

io_utils.py 内容：
- save_as_txt()
- save_as_srt()
- save_as_json()
- save_language_info()
- _format_timestamp()

models.py 内容：
- TranscriptionMetadata 类
- 其他数据结构定义
"""
