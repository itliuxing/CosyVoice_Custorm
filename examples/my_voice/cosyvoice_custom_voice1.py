#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CosyVoice3 自定义音色克隆脚本（优化版）

功能：
1. 使用参考音频进行 zero-shot 音色克隆
2. 自动拼接完整音频 chunk
3. 支持交互模式
4. 自动检查音频
5. 自动输出详细日志

推荐参考音频：
- wav 格式
- 16kHz
- 单声道
- 5~15 秒
- 无背景音乐
- 清晰人声

使用：
python cosyvoice_custom_voice.py

交互模式：
python cosyvoice_custom_voice.py --interactive
"""

import os
import sys
import torch
import torchaudio

from cosyvoice.cli.cosyvoice import AutoModel

# =========================================================
# 配置区域
# =========================================================

# 参考音频（建议 wav）
YOUR_VOICE_FILE = "./my_custom_voice_short.wav"

# 模型目录
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"

# 输出文件
OUTPUT_FILE = "./generated_speech.wav"

# 参考音频真实说的话
# 必须尽量和参考音频内容一致
PROMPT_TEXT = "大家好，今天测试一下 CosyVoice 的音色克隆效果。"

# 要生成的文本
SPEAK_TEXT = """
你好，这是一段使用克隆我本人音色生成的测试语音。
现在我们来测试一下模型的效果是否自然。
希望整个语音听起来更加自然、流畅，并且能够保持原始音色特征。
"""

# =========================================================
# 工具函数
# =========================================================

def check_audio_file(file_path):
    """检查音频文件"""

    if not os.path.exists(file_path):
        print(f"❌ 找不到音频文件: {file_path}")
        return False

    ext = os.path.splitext(file_path)[1].lower()

    if ext not in [".wav", ".mp3"]:
        print(f"❌ 不支持的音频格式: {ext}")
        return False

    try:
        info = torchaudio.info(file_path)

        print("\n📊 音频信息:")
        print(f"   文件: {file_path}")
        print(f"   采样率: {info.sample_rate}")
        print(f"   声道数: {info.num_channels}")

        duration = info.num_frames / info.sample_rate
        print(f"   时长: {duration:.2f} 秒")

        if duration > 30:
            print("❌ 参考音频超过 30 秒")
            return False

        if duration < 3:
            print("⚠️ 参考音频太短，效果可能较差")

        return True

    except Exception as e:
        print(f"❌ 音频读取失败: {e}")
        return False


def load_model():
    """加载 CosyVoice 模型"""

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


def generate_audio(
    cosyvoice,
    speak_text,
    output_file
):
    """
    生成语音
    """

    print("\n" + "=" * 60)
    print("🎤 开始生成语音")
    print("=" * 60)

    print(f"\n📝 Prompt 文本:")
    print(PROMPT_TEXT)

    print(f"\n📝 待生成文本:")
    print(speak_text)

    # CosyVoice3 必须包含 endofprompt token
    prompt = f"{PROMPT_TEXT}<|endofprompt|>{speak_text}"

    print("\n⚙️ 推理中，请稍候...")

    try:

        all_audio = []

        generator = cosyvoice.inference_zero_shot(
            speak_text,
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

			#if speech.shape[-1] == 0:
			#	continue

            print(f"✅ 收到音频 chunk: {idx}")

            all_audio.append(speech.cpu())

        # 没有生成任何音频
        if len(all_audio) == 0:
            print("❌ 没有生成任何音频数据")
            return False

        print(f"\n📦 共收到 {len(all_audio)} 个音频 chunk")

        # 拼接完整音频
        final_audio = torch.cat(all_audio, dim=1)

        print(f"📏 最终音频 shape: {final_audio.shape}")

        # 保存
        torchaudio.save(
            output_file,
            final_audio,
            cosyvoice.sample_rate
        )

        print("\n" + "=" * 60)
        print("✅ 语音生成完成")
        print("=" * 60)

        print(f"📁 输出文件: {output_file}")
        print(f"📊 采样率: {cosyvoice.sample_rate}")

        duration = final_audio.shape[1] / cosyvoice.sample_rate
        print(f"⏱️ 音频时长: {duration:.2f} 秒")

        return True

    except Exception as e:

        print("\n❌ 语音生成失败")
        print(f"错误信息: {e}")

        import traceback
        traceback.print_exc()

        return False


def interactive_mode():
    """
    交互模式
    """

    print("\n" + "=" * 60)
    print("🎯 进入交互模式")
    print("=" * 60)

    cosyvoice = load_model()

    if cosyvoice is None:
        return

    while True:

        print("\n" + "-" * 60)

        text = input("请输入要生成的文本（输入 quit 退出）:\n> ").strip()

        if text.lower() == "quit":
            print("👋 再见")
            break

        if not text:
            print("⚠️ 文本不能为空")
            continue

        output_file = f"./output_{len(text)}.wav"

        success = generate_audio(
            cosyvoice,
            text,
            output_file
        )

        if success:
            print(f"🎉 已生成: {output_file}")


def main():

    print("=" * 60)
    print("CosyVoice3 自定义音色克隆")
    print("=" * 60)

    # 检查音频
    if not check_audio_file(YOUR_VOICE_FILE):
        sys.exit(1)

    # 加载模型
    cosyvoice = load_model()

    if cosyvoice is None:
        sys.exit(1)

    # 交互模式
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":

        interactive_mode()

    else:

        success = generate_audio(
            cosyvoice,
            SPEAK_TEXT,
            OUTPUT_FILE
        )

        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()