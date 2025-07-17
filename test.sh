#!/bin/bash
# ============================================================================
# Whisper 模块测试脚本
# 用于快速测试音频和视频文件的转写功能
# ============================================================================

# 设置颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ============================================================================
# 环境检查
# ============================================================================
print_info "开始 Whisper 模块测试..."
echo ""

# 检查 whisper_tools 目录是否存在
if [ ! -d "whisper_tools" ]; then
    print_error "whisper_tools 目录不存在！"
    print_error "请确保在正确的项目根目录下运行此脚本。"
    exit 1
fi

# 检查必需的 Python 脚本是否存在
if [ ! -f "whisper_tools/transcribe_file.py" ]; then
    print_error "找不到 whisper_tools/transcribe_file.py！"
    exit 1
fi

if [ ! -f "whisper_tools/transcribe_batch.py" ]; then
    print_warning "找不到 whisper_tools/transcribe_batch.py（批量处理脚本）"
fi

# 检查 Python 环境
# 优先使用当前环境的 python，其次是 python3
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    print_error "找不到 python 或 python3 命令！"
    exit 1
fi

print_info "使用 Python 命令: $PYTHON_CMD"

print_success "环境检查通过"
echo ""

# ============================================================================
# 准备测试文件和目录
# ============================================================================
print_info "准备测试环境..."

# 创建必要的目录
mkdir -p audio media output

# 检查测试文件是否存在，如果不存在则创建示例文件
if [ ! -f "audio/test.mp3" ]; then
    print_warning "audio/test.mp3 不存在，请手动添加测试音频文件"
    # 创建一个占位文件用于演示
    touch "audio/test.mp3"
fi

if [ ! -f "media/example.mp4" ]; then
    print_warning "media/example.mp4 不存在，请手动添加测试视频文件"
    # 创建一个占位文件用于演示
    touch "media/example.mp4"
fi

# 清理之前的输出（可选）
print_info "清理之前的输出文件..."
rm -f output/test.txt output/example.srt output/example.txt

echo ""

# ============================================================================
# 测试 1: 音频文件转写（输出 TXT）
# ============================================================================
print_info "========== 测试 1: 音频文件转写 =========="
print_info "处理文件: audio/test.mp3"
print_info "输出格式: TXT"

if [ -f "audio/test.mp3" ] && [ -s "audio/test.mp3" ]; then
    # 执行转写
    $PYTHON_CMD whisper_tools/transcribe_file.py audio/test.mp3 --output-dir output --formats txt
    
    # 检查结果
    if [ -f "output/test.txt" ]; then
        print_success "音频转写成功！输出文件: output/test.txt"
        # 显示前几行内容
        echo -e "${BLUE}转写内容预览:${NC}"
        head -n 5 output/test.txt 2>/dev/null || echo "（文件为空或无法读取）"
    else
        print_error "音频转写失败！未生成 output/test.txt"
    fi
else
    print_warning "跳过测试 1：audio/test.mp3 文件不存在或为空"
fi

echo ""

# ============================================================================
# 测试 2: 视频文件转写（输出 SRT）
# ============================================================================
print_info "========== 测试 2: 视频文件转写 =========="
print_info "处理文件: media/example.mp4"
print_info "输出格式: SRT (字幕)"

if [ -f "media/example.mp4" ] && [ -s "media/example.mp4" ]; then
    # 执行转写
    $PYTHON_CMD whisper_tools/transcribe_file.py media/example.mp4 --output-dir output --formats srt
    
    # 检查结果
    if [ -f "output/example.srt" ]; then
        print_success "视频转写成功！输出文件: output/example.srt"
        # 显示前几行内容
        echo -e "${BLUE}字幕内容预览:${NC}"
        head -n 10 output/example.srt 2>/dev/null || echo "（文件为空或无法读取）"
    else
        print_error "视频转写失败！未生成 output/example.srt"
    fi
else
    print_warning "跳过测试 2：media/example.mp4 文件不存在或为空"
fi

echo ""

# ============================================================================
# 额外测试: 批量处理（如果 transcribe_batch.py 存在）
# ============================================================================
if [ -f "whisper_tools/transcribe_batch.py" ]; then
    print_info "========== 额外测试: 批量处理 =========="
    print_info "扫描 audio/ 和 media/ 目录"
    
    # 创建测试目录结构
    mkdir -p batch_output
    
    # 运行批量处理 - 方式1：直接运行脚本
    $PYTHON_CMD whisper_tools/transcribe_batch.py --input audio --output batch_output --formats txt
    
    # 也可以使用方式2：通过模块方式运行（需要设置 PYTHONPATH）
    # PYTHONPATH=. $PYTHON_CMD -m whisper_tools.transcribe_batch --input audio --output batch_output --formats txt
    
    # 检查结果
    if [ -n "$(ls -A batch_output 2>/dev/null)" ]; then
        print_success "批量处理完成！结果保存在 batch_output/ 目录"
        echo -e "${BLUE}批量处理结果:${NC}"
        find batch_output -type f -name "*.txt" -o -name "*.srt" | head -10
    else
        print_warning "批量处理未生成任何文件"
    fi
fi

echo ""

# ============================================================================
# 测试总结
# ============================================================================
print_info "========== 测试总结 =========="

# 统计生成的文件
txt_count=$(find output -name "*.txt" 2>/dev/null | wc -l)
srt_count=$(find output -name "*.srt" 2>/dev/null | wc -l)

echo "生成的文件统计："
echo "  - TXT 文件: $txt_count 个"
echo "  - SRT 文件: $srt_count 个"
echo ""

# 显示所有输出文件
if [ -n "$(ls -A output 2>/dev/null)" ]; then
    echo "输出文件列表："
    ls -la output/
else
    print_warning "output 目录为空"
fi

echo ""
print_success "测试脚本执行完成！"
echo ""

# ============================================================================
# 使用提示
# ============================================================================
echo -e "${YELLOW}使用提示:${NC}"
echo "1. 如果测试文件不存在，请手动添加："
echo "   - 音频文件: audio/test.mp3"
echo "   - 视频文件: media/example.mp4"
echo ""
echo "2. 可以使用以下命令单独运行转写："
echo "   $PYTHON_CMD whisper_tools/transcribe_file.py <文件路径> --formats txt srt"
echo ""
echo "3. 批量处理整个目录："
echo "   方式1: $PYTHON_CMD whisper_tools/transcribe_batch.py -i <输入目录> -o <输出目录>"
echo "   方式2: PYTHONPATH=. $PYTHON_CMD -m whisper_tools.transcribe_batch -i <输入目录> -o <输出目录>"
echo ""

# 设置脚本可执行权限提示
if [ ! -x "$0" ]; then
    echo -e "${YELLOW}提示: 运行 'chmod +x test.sh' 使脚本可执行${NC}"
fi