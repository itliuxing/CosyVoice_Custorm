# -*- coding: utf-8 -*-
#!/usr/bin/env python3
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
from tools.date.Date_Utils import get_now_str_code
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
out_file = f"{get_now_str_code()}.wav"
OUTPUT_FILE_PATH = os.path.join(project_root_path, 'asset/generator_out_path', out_file)

# 每句输出目录
SEGMENT_DIR = os.path.join(project_root_path, 'asset/generator_out_path/segments')

# 参考音频真实文本
# 必须和参考音频内容尽量一致
PROMPT_TEXT = "大家好，今天测试一下 CosyVoice 的音色克隆效果。"

# 官方推荐 system prompt
SYSTEM_PROMPT = "You are a helpful assistant."

# 要生成的文本
SPEAK_TEXT = """
你好，这是一段使用克隆我本人音色生成的测试语音。
现在我们来测试一下模型的效果是否自然。
希望整个语音听起来更加自然流畅。
"""


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


def generate_full_audio(cosyvoiceAutoModel):

    print("\n" + "=" * 60)
    print("🎯 开始生成完整语音")
    print("=" * 60)

    # 构建 prompt_text（与 example.py 中 cosyvoice3_example 的 zero_shot 格式一致）
    prompt_text = f"{SYSTEM_PROMPT}<|endofprompt|>{PROMPT_TEXT}"
    print(f"\n📝 Prompt: {prompt_text}")

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_FILE_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")

    # 收集所有音频 chunk
    all_chunks = []

    # 使用 inference_zero_shot 进行音色克隆
    # 格式：inference_zero_shot(tts_text, prompt_text, prompt_wav, zero_shot_spk_id='', stream=False)
    cosyvoice_generator = cosyvoiceAutoModel.inference_zero_shot(SPEAK_TEXT, prompt_text, YOUR_VOICE_FILE, stream=False)

    for i, result in enumerate(cosyvoice_generator):
        speech = result['tts_speech']
        print(f"✅ chunk {i}: shape={speech.shape}")
        all_chunks.append(speech.cpu())

    if len(all_chunks) == 0:
        print("❌ 没有生成任何音频")
        return False

    # 合并所有 chunk
    final_audio = torch.cat(all_chunks, dim=1)

    # 保存到 OUTPUT_FILE_PATH
    torchaudio.save(OUTPUT_FILE_PATH, final_audio, cosyvoiceAutoModel.sample_rate)

    duration = final_audio.shape[1] / cosyvoiceAutoModel.sample_rate

    print("\n" + "=" * 60)
    print("✅ 完整语音生成成功")
    print("=" * 60)

    print(f"📁 输出文件: {OUTPUT_FILE_PATH}")
    print(f"📊 采样率: {cosyvoiceAutoModel.sample_rate}")
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