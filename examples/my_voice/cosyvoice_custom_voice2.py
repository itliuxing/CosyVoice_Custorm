#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CosyVoice3 自定义音色克隆（稳定版）

特点：
1. 使用官方推荐协议
2. 自动拼接完整音频
3. 避免只生成半句
4. 避免空音频
5. 统一使用 spaces 缩进（不会 TabError）

运行：
python cosyvoice_custom_voice2.py
"""

import os
import sys
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel
from tools.files.File_Path_Utils import get_project_root

# =========================================================
# 配置区域
# =========================================================


project_root_path = get_project_root()

# 参考音频  本处为：项目根目录 + resource + 配置文件
config_file = "my_custom_voice_short.wav"
YOUR_VOICE_FILE = os.path.join(project_root_path, 'asset', config_file)

# 模型目录
model_file = "Fun-CosyVoice3-0.5B"
MODEL_DIR = os.path.join(project_root_path, "pretrained_models" , model_file)

# 输出文件
out_file = "generated_speech.wav"
OUTPUT_FILE_PATH = os.path.join(project_root_path, 'asset/generator_out_path', out_file)

# 参考音频真实文本
# 必须和参考音频内容尽量一致
PROMPT_TEXT = "大家好，今天测试一下 CosyVoice 的音色克隆效果。"

# 要生成的文本
SPEAK_TEXT = """
你好，这是一段使用克隆我本人音色生成的测试语音。
现在我们来测试一下模型的效果是否自然。
希望整个语音听起来更加自然流畅。
"""

# 官方推荐 system prompt
SYSTEM_PROMPT = "You are a helpful assistant."


# =========================================================
# 工具函数
# =========================================================

def check_audio_file(file_path):

    if not os.path.exists(file_path):
        print(f"❌ 找不到音频文件: {file_path}")
        return False

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in [".wav", ".mp3"]:
        print(f"❌ 不支持的音频格式: {ext}")
        return False

    try:
        info = torchaudio.info(file_path)

        duration = info.num_frames / info.sample_rate

        print("\n📊 音频信息")
        print("=" * 50)
        print(f"文件: {file_path}")
        print(f"采样率: {info.sample_rate}")
        print(f"声道数: {info.num_channels}")
        print(f"时长: {duration:.2f} 秒")

        if duration > 30:
            print("❌ 参考音频超过 30 秒")
            return False

        return True

    except Exception as e:
        print(f"❌ 音频读取失败: {e}")
        return False


def split_text(text):
    """
    长文本分句
    """

    text = text.replace("\n", "")

    separators = ["。", "！", "？", ".", "!", "?"]

    result = []
    current = ""

    for char in text:

        current += char

        if char in separators:

            current = current.strip()

            if current:
                result.append(current)

            current = ""

    current = current.strip()

    if current:
        result.append(current)

    return result


def load_model():

    print("\n📥 正在加载模型...")
    print(f"模型目录: {MODEL_DIR}")

    try:

        cosyvoice = AutoModel(
            model_dir=MODEL_DIR
        )

        print("✅ 模型加载成功")

        return cosyvoice

    except Exception as e:

        print(f"❌ 模型加载失败: {e}")

        return None


def generate_segment(
    cosyvoice,
    text
):
    """
    生成单句音频
    """

    print("\n" + "-" * 50)
    print(f"🎤 当前生成文本:")
    print(text)

    # 官方协议
    prompt = f"{SYSTEM_PROMPT}<|endofprompt|>{PROMPT_TEXT}"

    print("\n📝 Prompt:")
    print(prompt)

    all_chunks = []

    try:

        generator = cosyvoice.inference_zero_shot(
            text,
            prompt,
            YOUR_VOICE_FILE,
            stream=False
        )

        for idx, result in enumerate(generator):

            if result is None:
                continue

            if 'tts_speech' not in result:
                continue

            speech = result['tts_speech']

            if speech is None:
                continue

            if speech.shape[-1] == 0:
                continue

            print(f"✅ chunk {idx}: {speech.shape}")

            all_chunks.append(speech.cpu())

        if len(all_chunks) == 0:
            print("❌ 当前句子没有生成音频")
            return None

        final_segment = torch.cat(all_chunks, dim=1)

        print(f"📏 当前句子最终 shape: {final_segment.shape}")

        return final_segment

    except Exception as e:

        print(f"❌ 当前句子生成失败: {e}")

        import traceback
        traceback.print_exc()

        return None


def generate_full_audio(cosyvoice):

    print("\n" + "=" * 60)
    print("🎯 开始生成完整语音")
    print("=" * 60)

    text_segments = split_text(SPEAK_TEXT)

    print("\n📝 分句结果:")

    for idx, seg in enumerate(text_segments):
        print(f"{idx+1}. {seg}")

    all_segments = []

    for idx, text in enumerate(text_segments):

        print("\n" + "=" * 60)
        print(f"🚀 正在生成第 {idx+1}/{len(text_segments)} 段")
        print("=" * 60)

        segment_audio = generate_segment(
            cosyvoice,
            text
        )

        if segment_audio is not None:
            all_segments.append(segment_audio)

    if len(all_segments) == 0:

        print("❌ 没有生成任何有效音频")

        return False

    print("\n📦 正在拼接所有音频段...")

    final_audio = torch.cat(all_segments, dim=1)

    print(f"📏 最终音频 shape: {final_audio.shape}")

    torchaudio.save(
        OUTPUT_FILE_PATH,
        final_audio,
        cosyvoice.sample_rate
    )

    duration = final_audio.shape[1] / cosyvoice.sample_rate

    print("\n" + "=" * 60)
    print("✅ 完整语音生成成功")
    print("=" * 60)

    print(f"📁 输出文件: {OUTPUT_FILE_PATH}")
    print(f"📊 采样率: {cosyvoice.sample_rate}")
    print(f"⏱️ 音频时长: {duration:.2f} 秒")

    return True


def main():

    print("=" * 60)
    print("CosyVoice3 自定义音色克隆")
    print("=" * 60)

    if not check_audio_file(YOUR_VOICE_FILE):
        sys.exit(1)

    cosyvoice = load_model()

    if cosyvoice is None:
        sys.exit(1)

    success = generate_full_audio(cosyvoice)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()