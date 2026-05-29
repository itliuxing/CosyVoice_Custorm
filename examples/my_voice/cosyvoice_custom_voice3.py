# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
CosyVoice3 自定义音色克隆（整合版）

结合 cosyvoice_custom_voice2.py 和 example.py 的成功模式

运行：
python cosyvoice_custom_voice3.py
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

# 参考音频
config_file = "my_custom_voice_short.wav"
YOUR_VOICE_FILE = os.path.join(project_root_path, 'asset', config_file)

# 模型目录
model_file = "Fun-CosyVoice3-0.5B"
MODEL_DIR = os.path.join(project_root_path, "pretrained_models" , model_file)

# 输出文件
out_file = "generated_speech.wav"
OUTPUT_FILE_PATH = os.path.join(project_root_path, 'asset/generator_out_path', out_file)

# 参考音频真实文本（必须和参考音频内容一致）
PROMPT_TEXT = "大家好，今天测试一下 CosyVoice 的音色克隆效果。"

# 要生成的文本
SPEAK_TEXT = """
你好，这是一段使用克隆我本人音色生成的测试语音。
现在我们来测试一下模型的效果是否自然。
希望整个语音听起来更加自然流畅。
"""

# System prompt
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


def generate_single_audio(cosyvoice, tts_text, prompt_text, prompt_wav, output_file):
    """
    参考 example.py 中 cosyvoice3_example() 的成功模式
    生成单个音频文件
    """

    print("\n" + "-" * 50)
    print(f"🎤 待生成文本: {tts_text}")

    try:
        all_chunks = []

        generator = cosyvoice.inference_zero_shot(
            tts_text,
            prompt_text,
            prompt_wav,
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

            print(f"✅ chunk {idx}: shape={speech.shape}")

            all_chunks.append(speech.cpu())

        if len(all_chunks) == 0:
            print("❌ 没有生成任何音频")
            return False

        final_audio = torch.cat(all_chunks, dim=1)

        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        torchaudio.save(output_file, final_audio, cosyvoice.sample_rate)

        duration = final_audio.shape[1] / cosyvoice.sample_rate
        print(f"💾 已保存: {output_file} (时长: {duration:.2f} 秒)")

        return True

    except Exception as e:

        print(f"❌ 生成失败: {e}")

        import traceback
        traceback.print_exc()

        return False


def generate_full_audio(cosyvoice):

    print("\n" + "=" * 60)
    print("🎯 开始生成完整语音")
    print("=" * 60)

    # 构建 prompt_text（参考 example.py 中的成功格式）
    prompt_text = f"{SYSTEM_PROMPT}<|endofprompt|>{PROMPT_TEXT}"
    print(f"\n📝 Prompt: {prompt_text}")

    # 分句
    text_segments = split_text(SPEAK_TEXT)

    print("\n📝 分句结果:")
    for idx, seg in enumerate(text_segments):
        print(f"  {idx+1}. {seg}")

    # 单独生成每句
    segment_files = []

    for idx, text in enumerate(text_segments):

        print("\n" + "=" * 60)
        print(f"🚀 正在生成第 {idx+1}/{len(text_segments)} 段")
        print("=" * 60)

        segment_file = os.path.join(
            project_root_path,
            'asset/generator_out_path/segments',
            f'segment_{idx:02d}.wav'
        )

        success = generate_single_audio(
            cosyvoice,
            text,
            prompt_text,
            YOUR_VOICE_FILE,
            segment_file
        )

        if success:
            segment_files.append(segment_file)

    if len(segment_files) == 0:
        print("❌ 没有生成任何有效音频")
        return False

    # 合并所有音频
    print("\n" + "=" * 60)
    print("📦 合并所有音频段...")
    print("=" * 60)

    all_audio = []

    for segment_file in segment_files:

        try:
            audio, sample_rate = torchaudio.load(segment_file)
            all_audio.append(audio)
            duration = audio.shape[1] / sample_rate
            print(f"  加载: {segment_file} (时长: {duration:.2f} 秒)")
        except Exception as e:
            print(f"  ⚠️ 加载失败 {segment_file}: {e}")

    if len(all_audio) == 0:
        print("❌ 没有可合并的音频")
        return False

    final_audio = torch.cat(all_audio, dim=1)

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    torchaudio.save(OUTPUT_FILE_PATH, final_audio, cosyvoice.sample_rate)

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
    print("CosyVoice3 自定义音色克隆（整合版）")
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