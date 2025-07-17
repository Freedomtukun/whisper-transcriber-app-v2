#!/usr/bin/env python3
"""
批量处理音频和视频文件的 Whisper 转写脚本。

该脚本可以递归扫描指定目录下的所有音频和视频文件，
使用 Whisper 进行转写，并保存为多种格式。
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# 导入已有的转写模块
from whisper_tools.transcribe_file import (
    transcribe_file, 
    save_as_txt, 
    save_as_srt, 
    save_as_json,
    is_audio,
    is_video
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 🔍 文件扫描与过滤
# ---------------------------------------------------------------------------
def find_media_files(
    input_dir: Path, 
    recursive: bool = False,
    extensions: Set[str] = None
) -> List[Path]:
    """
    扫描目录中的音频和视频文件。
    
    参数:
        input_dir: 输入目录
        recursive: 是否递归扫描子目录
        extensions: 指定的文件扩展名集合（可选）
    
    返回:
        符合条件的文件路径列表
    """
    pattern = "**/*" if recursive else "*"
    media_files = []
    
    for file_path in input_dir.glob(pattern):
        if file_path.is_file():
            # 检查是否为音频或视频文件
            if is_audio(str(file_path)) or is_video(str(file_path)):
                # 如果指定了扩展名，进一步过滤
                if extensions is None or file_path.suffix.lower() in extensions:
                    media_files.append(file_path)
    
    return sorted(media_files)

# ---------------------------------------------------------------------------
# 🎯 单文件处理
# ---------------------------------------------------------------------------
def process_single_file(
    input_file: Path,
    input_base: Path,
    output_base: Path,
    formats: List[str],
    language: str = None,
    force: bool = False
) -> Tuple[bool, str]:
    """
    处理单个文件的转写。
    
    参数:
        input_file: 输入文件路径
        input_base: 输入基础目录（用于计算相对路径）
        output_base: 输出基础目录
        formats: 输出格式列表
        language: 指定语言
        force: 是否强制覆盖已存在的文件
    
    返回:
        (成功与否, 消息)
    """
    try:
        # 计算相对路径和输出路径
        relative_path = input_file.relative_to(input_base)
        output_dir = output_base / relative_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否需要跳过
        base_name = input_file.stem
        skip_file = False
        
        if not force:
            existing_files = []
            if "txt" in formats and (output_dir / f"{base_name}.txt").exists():
                existing_files.append("txt")
            if "srt" in formats and (output_dir / f"{base_name}.srt").exists():
                existing_files.append("srt")
            if "json" in formats and (output_dir / f"{base_name}.json").exists():
                existing_files.append("json")
            
            if len(existing_files) == len(formats):
                skip_file = True
                return True, f"跳过 {relative_path} (输出文件已存在)"
        
        # 执行转写
        logger.info(f"正在处理: {relative_path}")
        start_time = time.time()
        
        text, segments = transcribe_file(str(input_file), language=language)
        
        # 保存结果
        saved_formats = []
        
        if "txt" in formats:
            txt_path = output_dir / f"{base_name}.txt"
            save_as_txt(text, str(txt_path))
            saved_formats.append("txt")
        
        if "srt" in formats:
            srt_path = output_dir / f"{base_name}.srt"
            save_as_srt(segments, str(srt_path))
            saved_formats.append("srt")
        
        if "json" in formats:
            json_path = output_dir / f"{base_name}.json"
            save_as_json({"text": text, "segments": segments}, str(json_path))
            saved_formats.append("json")
        
        elapsed_time = time.time() - start_time
        return True, f"完成 {relative_path} [{', '.join(saved_formats)}] (耗时: {elapsed_time:.1f}秒)"
        
    except Exception as e:
        return False, f"失败 {relative_path}: {str(e)}"

# ---------------------------------------------------------------------------
# 🚀 批量处理主函数
# ---------------------------------------------------------------------------
def batch_transcribe(
    input_dir: str,
    output_dir: str,
    formats: List[str] = None,
    recursive: bool = False,
    language: str = None,
    force: bool = False,
    max_workers: int = 1,
    extensions: List[str] = None
) -> None:
    """
    批量转写音频和视频文件。
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        formats: 输出格式列表，默认 ['txt', 'srt']
        recursive: 是否递归扫描子目录
        language: 指定语言代码
        force: 是否强制覆盖已存在的文件
        max_workers: 并发处理的最大线程数
        extensions: 限定的文件扩展名列表
    """
    # 默认输出格式
    if formats is None:
        formats = ['txt', 'srt']
    
    # 路径处理
    input_path = Path(input_dir).resolve()
    output_path = Path(output_dir).resolve()
    
    # 检查输入目录
    if not input_path.exists():
        logger.error(f"输入目录不存在: {input_path}")
        sys.exit(1)
    
    if not input_path.is_dir():
        logger.error(f"输入路径不是目录: {input_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 转换扩展名为集合
    ext_set = None
    if extensions:
        ext_set = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                   for ext in extensions}
    
    # 扫描文件
    logger.info(f"扫描目录: {input_path}")
    media_files = find_media_files(input_path, recursive, ext_set)
    
    if not media_files:
        logger.warning("未找到任何音频或视频文件")
        return
    
    logger.info(f"找到 {len(media_files)} 个文件待处理")
    logger.info(f"输出格式: {', '.join(formats)}")
    logger.info(f"输出目录: {output_path}")
    
    # 统计信息
    success_count = 0
    skip_count = 0
    fail_count = 0
    
    # 处理文件
    if max_workers == 1:
        # 单线程处理
        for media_file in media_files:
            success, message = process_single_file(
                media_file, input_path, output_path, 
                formats, language, force
            )
            logger.info(message)
            
            if success:
                if "跳过" in message:
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
    else:
        # 多线程处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(
                    process_single_file,
                    media_file, input_path, output_path,
                    formats, language, force
                ): media_file
                for media_file in media_files
            }
            
            # 处理完成的任务
            for future in as_completed(future_to_file):
                success, message = future.result()
                logger.info(message)
                
                if success:
                    if "跳过" in message:
                        skip_count += 1
                    else:
                        success_count += 1
                else:
                    fail_count += 1
    
    # 输出统计
    logger.info("=" * 50)
    logger.info(f"处理完成! 成功: {success_count}, 跳过: {skip_count}, 失败: {fail_count}")
    logger.info(f"总计: {len(media_files)} 个文件")

# ---------------------------------------------------------------------------
# 🎮 命令行接口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="批量转写音频和视频文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础用法
  python transcribe_batch.py --input media/ --output output/
  
  # 递归扫描并指定格式
  python transcribe_batch.py -i media/ -o output/ -r --formats txt srt json
  
  # 指定语言和文件类型
  python transcribe_batch.py -i audio/ -o transcripts/ --lang zh --ext mp3 wav
  
  # 多线程处理
  python transcribe_batch.py -i videos/ -o output/ --workers 4
        """
    )
    
    # 必需参数
    parser.add_argument('-i', '--input', required=True,
                        help='输入目录路径')
    parser.add_argument('-o', '--output', required=True,
                        help='输出目录路径')
    
    # 可选参数
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='递归扫描子目录')
    parser.add_argument('--formats', nargs='+', 
                        choices=['txt', 'srt', 'json'],
                        default=['txt', 'srt'],
                        help='输出格式 (默认: txt srt)')
    parser.add_argument('--lang', '--language',
                        help='指定语言代码 (如: zh, en, ja)')
    parser.add_argument('-f', '--force', action='store_true',
                        help='强制覆盖已存在的输出文件')
    parser.add_argument('--ext', '--extensions', nargs='+',
                        help='限定文件扩展名 (如: mp3 wav mp4)')
    parser.add_argument('-w', '--workers', type=int, default=1,
                        help='并发处理的线程数 (默认: 1)')
    
    args = parser.parse_args()
    
    # 执行批量处理
    try:
        batch_transcribe(
            input_dir=args.input,
            output_dir=args.output,
            formats=args.formats,
            recursive=args.recursive,
            language=args.lang,
            force=args.force,
            max_workers=args.workers,
            extensions=args.ext
        )
    except KeyboardInterrupt:
        logger.info("\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"批量处理失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()