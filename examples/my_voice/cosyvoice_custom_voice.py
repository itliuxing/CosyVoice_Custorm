#!/usr/bin/env python3
"""
CosyVoice 自定义音色克隆脚本
使用你自己的音频文件作为参考音色，生成对应音色的语音

使用方法：
1. 将你的音频文件放在 ./my_custom_voice.wav 位置（支持 wav, mp3 格式）
2. 修改 SPEAK_TEXT 为你想要生成的文本内容
3. 运行脚本：python cosyvoice_custom_voice.py

注意：
- 参考音频建议5-30秒，语音清晰，无背景音乐
- 音频会被用于克隆音色特征
"""

import os
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel

# ============ 配置参数 ============

# 你的音色参考音频文件路径（支持 wav, mp3 格式）
# 请确保这个文件存在且音频清晰
YOUR_VOICE_FILE = "./my_custom_voice_short.wav"

# 预训练模型路径（请根据实际情况修改）
MODEL_DIR = "pretrained_models/Fun-CosyVoice3-0.5B"

# 生成的文本内容（你可以修改这里）
# 这是要使用你的音色说出的内容
SPEAK_TEXT = "你好，这是一段使用克隆我本人音色生成的测试语音。现在我们来测试一下模型的效果是否自然。"

# Prompt 文本（通常和参考音频的内容相关）
# 这个文本应该描述语音的风格或内容
PROMPT_TEXT = "大家好，今天测试一下 CosyVoice 的音色克隆效果。"

# 生成的音频保存路径
OUTPUT_FILE = "./generated_speech.wav"


def check_audio_file(file_path):
    """检查音频文件是否存在且格式正确"""
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到音频文件 '{file_path}'")
        print(f"\n请将你的音频文件重命名为 'my_custom_voice.wav' 并放在当前目录下")
        print(f"或者修改代码中的 YOUR_VOICE_FILE 变量指向你的音频文件")
        return False

    # 检查文件扩展名
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in ['.wav', '.mp3']:
        print(f"❌ 错误：不支持的文件格式 '{ext}'")
        print(f"仅支持 .wav 和 .mp3 格式")
        return False

    print(f"✅ 找到音频文件：{file_path}")
    return True


def generate_speech_with_custom_voice():
    """使用自定义音色生成语音"""
    print("=" * 50)
    print("CosyVoice 自定义音色克隆")
    print("=" * 50)

    # 1. 检查音频文件
    if not check_audio_file(YOUR_VOICE_FILE):
        return False

    # 2. 加载模型
    print(f"\n📥 正在加载模型：{MODEL_DIR}")
    print("（首次运行可能需要几分钟下载模型）...")

    try:
        cosyvoice = AutoModel(model_dir=MODEL_DIR)
        print("✅ 模型加载成功！")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        print("\n请确保：")
        print("1. 已安装 CosyVoice 库：pip install cosyvoice")
        print(f"2. 模型路径 '{MODEL_DIR}' 正确")
        print("3. 已下载预训练模型")
        return False

    # 3. 准备 prompt
    prompt = f"{PROMPT_TEXT}<|endofprompt|>{SPEAK_TEXT}"

    print(f"\n📝 生成参数：")
    print(f"   参考音频：{YOUR_VOICE_FILE}")
    print(f"   生成文本：{SPEAK_TEXT}")
    print(f"   Prompt：{PROMPT_TEXT}")

    # 4. 生成语音
    print("\n🎤 正在生成语音，请稍候...")

    try:
        # 使用 zero-shot 方法进行音色克隆
        # stream=False 表示等待生成完成再保存
        for i, result in enumerate(cosyvoice.inference_zero_shot(
            SPEAK_TEXT,  # 要生成的文本
            prompt,       # Prompt 文本
            YOUR_VOICE_FILE,  # 参考音频（用于克隆音色）
            stream=False
        )):
            # 保存生成的语音
            torchaudio.save(OUTPUT_FILE, result['tts_speech'], cosyvoice.sample_rate)
            print(f"\n✅ 语音生成成功！")
            print(f"📁 保存位置：{OUTPUT_FILE}")
            print(f"📊 采样率：{cosyvoice.sample_rate} Hz")
            return True

    except Exception as e:
        print(f"❌ 语音生成失败：{e}")
        return False


def interactive_mode():
    """交互模式：让用户多次输入不同文本"""
    print("\n" + "=" * 50)
    print("进入交互模式")
    print("=" * 50)
    print("输入你想要用你的音色说的话，输入 'quit' 退出")

    # 加载模型（只加载一次）
    print(f"\n📥 正在加载模型...")
    try:
        cosyvoice = AutoModel(model_dir=MODEL_DIR)
        print("✅ 模型加载成功！\n")
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return

    while True:
        print("-" * 40)
        text = input("请输入文本（输入 'quit' 退出）: ").strip()

        if text.lower() == 'quit':
            print("再见！")
            break

        if not text:
            print("⚠️ 请输入有效的文本")
            continue

        prompt = f"{PROMPT_TEXT}<|endofprompt|>{text}"
        output_name = f"./output_{len(text)}.wav"

        print(f"🎤 正在生成...")
        try:
            for i, result in enumerate(cosyvoice.inference_zero_shot(
                text,
                prompt,
                YOUR_VOICE_FILE,
                stream=False
            )):
                torchaudio.save(output_name, result['tts_speech'], cosyvoice.sample_rate)
                print(f"✅ 已保存到：{output_name}")
        except Exception as e:
            print(f"❌ 生成失败：{e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        # 交互模式
        if not check_audio_file(YOUR_VOICE_FILE):
            sys.exit(1)
        interactive_mode()
    else:
        # 单次生成模式
        success = generate_speech_with_custom_voice()
        sys.exit(0 if success else 1)
