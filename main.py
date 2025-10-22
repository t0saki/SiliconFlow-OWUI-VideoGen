"""
title: SiliconFlow Video Pipeline
author: tosaki
author_url: https://github.com/t0saki
project_url: (Your Project URL)
version: 1.0.0
license: Apache License 2.0
description:
  一个使用 SiliconFlow API 生成视频的管道。它能根据用户的多模态输入自动在 T2V (文本生成视频) 和 I2V (图像生成视频) 模型之间智能切换。/A pipeline for generating videos using the SiliconFlow API. It automatically switches between Text-to-Video (T2V) and Image-to-Video (I2V) models based on user input.
features:
  en:
    - Automatic detection of image input to switch between Text-to-Video (T2V) and Image-to-Video (I2V) models.
    - Asynchronous API calls (aiohttp) for non-blocking video generation.
    - Efficient status polling mechanism with configurable check intervals and timeouts.
    - Provides progressive status updates ("Generating...", "Task ID...", "Success/Failure") to the user.
    - Highly configurable via Pydantic settings (API Key, model names, image size, negative prompt, seed).
    - Gracefully handles cases where an I2V model is not configured, falling back to T2V.
    - Parses the last user message to extract multimodal content (text and images).
    - Formats the final output as an embedded HTML <video> player and a markdown download link.
    - Robust error handling for API failures, HTTP errors, and timeouts.
  zh:
    - 自动检测用户输入中的图像，智能切换 T2V (文本生成视频) 和 I2V (图像生成视频) 模型。
    - 使用 aiohttp 实现异步 API 调用，实现非阻塞式视频生成。
    - 高效的状态轮询机制，支持自定义检查间隔和超时时间。
    - 为用户提供渐进式状态更新 (如 "生成中...", "任务ID...", "成功/失败")，提升用户体验。
    - 通过 Pydantic (Valves) 提供高度可配置的设置 (API 密钥, 模型名称, 图像尺寸, 负面提示词, 种子)。
    - 优雅处理未配置 I2V 模型的场景，自动回退到 T2V 模式并忽略图像。
    - 自动解析最后一条用户消息以提取多模态内容 (文本和图像 URL)。
    - 将最终输出格式化为 HTML 嵌入式 <video> 播放器和 Markdown 下载链接。
    - 健壮的错误处理机制，覆盖 API 失败、HTTP 错误和请求超时。
"""

import asyncio
import aiohttp
import time
from pydantic import BaseModel, Field
from typing import List, Optional, Callable, Awaitable, AsyncGenerator


class Pipe:
    class Valves(BaseModel):
        siliconflow_API_KEY: str = Field(
            default="", description="API Key for SiliconFlow")
        # Modified: 拆分为 T2V 和 I2V 模型配置
        t2v_model: str = Field(
            default="Wan-AI/Wan2.2-T2V-A14B",
            description="文本生成视频 (T2V) 模型 (默认使用)"
        )
        i2v_model: Optional[str] = Field(
            default="Wan-AI/Wan2.2-I2V-A14B",
            description="图片生成视频 (I2V) 模型 (可选, 当有图片输入时自动使用)"
        )
        # --- 原有配置 ---
        image_size: str = Field(
            default="1280x720", description="视频尺寸 (Video dimensions: 1280x720, 720x1280, 960x960)")
        negative_prompt: Optional[str] = Field(
            default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走", description="负面提示词 (Negative prompt)")
        timeout: int = Field(
            default=300, description="超时时间 (Time out for SiliconFlow)")
        check_interval: int = Field(
            default=10, description="状态检查间隔 (Check interval for Video Status)")
        seed: int = Field(
            default=42, description="随机种子 (Seed for SiliconFlow)")
        StreamTime: float = Field(
            default=1.0, description="流式刷新时间 (Stream time just for Show)")

    def __init__(self):
        self.type = "manifold"
        self.name = "SiliconFlow: "
        self.valves = self.Valves()
        self.emitter = None
        pass

    async def emit_status(
        self,
        message: str = "",
        done: bool = False,
    ):
        if self.emitter:
            await self.emitter({
                "type": "status",
                "data": {
                    "description": message,
                    "done": done,
                },
            })


    def creat_video(self, video_url):
        vider_html = "```html\n<video style='width: 100%; max-width: 800px; height: auto;' controls='controls' autoplay='autoplay' loop='loop' preload='auto' src='{}'></video>\n```".format(
            video_url)
        video_download = "👉[点击下载视频]({})".format(video_url)
        return vider_html + "\n\n" + video_download

    async def get_requestId(
        self,
        pro: str,
        model: str,
        image_size: str,
        seed: int = 123,
        negative_prompt: Optional[str] = None,
        image: Optional[str] = None
    ):
        url = "https://api.siliconflow.cn/v1/video/submit"

        # 动态构建 payload
        payload = {
            "model": model,
            "prompt": pro,
            "image_size": image_size,
            "seed": seed
        }

        # 仅在提供了可选参数时才添加它们
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if image:
            payload["image"] = image

        headers = {
            "Authorization": f"Bearer {self.valves.siliconflow_API_KEY}",
            "Content-Type": "application/json",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=10) as response:
                response_text = await response.text()
                if response.status != 200:
                    raise Exception(
                        f"HTTP Error {response.status}: {response_text}")

                response_json = await response.json()
                if "requestId" not in response_json:
                    raise Exception(
                        f"API Error: 'requestId' not in response: {response_text}")
                return response_json["requestId"]

    async def get_video(self, requestId, session: aiohttp.ClientSession):
        url = "https://api.siliconflow.cn/v1/video/status"
        payload = {"requestId": requestId}
        headers = {
            "Authorization": f"Bearer {self.valves.siliconflow_API_KEY}",
            "Content-Type": "application/json",
        }
        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                raise Exception(f"HTTP Error {response.status}: {await response.text()}")

            response_json = await response.json()
            status = response_json.get("status", "Failed")  # 默认为 Failed 以防万一

            if status == "Succeed":
                # 确保 results 和 videos 列表存在
                videos = response_json.get("results", {}).get("videos", [])
                if videos:
                    return status, videos[-1].get("url", "")
                else:
                    # 成功但没有视频URL，这很奇怪，但我们应该处理它
                    return "Failed", "Succeeded but no video URL found."
            elif status == "Failed":
                return status, response_json.get("reason", "Unknown failure reason")

            # InQueue or InProgress
            return status, ""

    async def loop_get_video(self, requestId) -> AsyncGenerator[str, None]:
        await self.emit_status(message=f"任务id: {requestId}")
        yield "任务id: " + requestId

        start_time = time.time()
        position = 0

        async with aiohttp.ClientSession() as session:
            while True:
                spend_time = time.time() - start_time

                # 检查是否超时
                if spend_time > self.valves.timeout:
                    await self.emit_status(message=f"💥视频生成失败 (超时)", done=True)
                    yield str(self.valves.timeout) + "s Time out"
                    return

                # 只有在到达检查间隔时才真正查询
                if spend_time > position:
                    position += self.valves.check_interval

                    try:
                        status, url_or_reason = await self.get_video(requestId, session)

                        if status == "Succeed":
                            await self.emit_status(message="🎉视频生成成功", done=True)
                            yield "\n\n" + self.creat_video(url_or_reason)
                            return
                        elif status == "Failed":
                            await self.emit_status(message=f"💥视频生成失败: {url_or_reason}", done=True)
                            yield f"Error: Generation Failed. Reason: {url_or_reason}"
                            return
                        # 如果是 InQueue 或 InProgress，则不做任何事，继续循环

                    except Exception as e:
                        await self.emit_status(message=f"💥网络或API错误: {str(e)}", done=True)
                        yield f"Network/API Error: {str(e)}"
                        return

                # 在两次检查之间，模拟流式更新状态
                await self.emit_status(message=f"💤视频生成中，请等待几分钟...{spend_time:.1f}s")
                await asyncio.sleep(self.valves.StreamTime)  # 短暂休眠以避免CPU空转

    # Modified: 只暴露一个虚拟模型
    def pipes(self) -> List[dict]:
        return [
            {"id": "siliconflow-video-auto",
                "name": "SiliconFlow 视频生成 (自动 T2V/I2V)"}
        ]


    async def pipe(self, body: dict,
               *,  # <-- ADD this bare asterisk
               user: Optional[dict] = None,
               __event_emitter__: Callable[[dict], Awaitable[None]] = None,
               __event_call__: Callable[[dict], Awaitable[dict]] = None,
               ) -> AsyncGenerator[str, None]:

        self.emitter = __event_emitter__
        messages = body.get("messages", [])

        if not messages:
            await self.emit_status(message="💥错误: 未提供消息", done=True)
            yield "Error: No messages provided"
            return

        # --- 新逻辑：解析 prompt 和 image ---
        prompt = ""
        image = None

        # 寻找最后一条用户消息
        last_user_message = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_message = msg
                break

        if not last_user_message:
            await self.emit_status(message="💥错误: 未找到用户消息", done=True)
            yield "Error: No user message found"
            return

        content = last_user_message.get("content")

        if isinstance(content, str):
            prompt = content
        elif isinstance(content, list):
            # 处理多模态内容列表
            for part in content:
                if part.get("type") == "text":
                    prompt = part.get("text", "")
                elif part.get("type") == "image_url":
                    # API 接受 base64 数据 URI 或 http/https URL
                    image_url = part.get("image_url", {}).get("url")
                    if image_url:
                        image = image_url

        if not prompt:
            await self.emit_status(message="💥错误: 在最后一条用户消息中未找到文本提示", done=True)
            yield "Error: No text prompt found in the last user message"
            return
        # --- 消息解析结束 ---

        # --- Modified: 自动模型选择 ---
        model_to_use = self.valves.t2v_model  # 默认 T2V

        if image:
            if self.valves.i2v_model:
                # 场景 1: 有图片, 且配置了 I2V
                model_to_use = self.valves.i2v_model
                await self.emit_status(message=f"检测到图片输入，使用 I2V 模型: {model_to_use}")
            else:
                # 场景 2: 有图片, 但未配置 I2V
                await self.emit_status(message=f"检测到图片输入，但未配置 I2V 模型。将忽略图片，使用 T2V 模型: {self.valves.t2v_model}")
                image = None  # 强制 T2V 模型不接收图片
                model_to_use = self.valves.t2v_model
        else:
            # 场景 3: 没有图片, 使用 T2V
            await self.emit_status(message=f"使用 T2V 模型: {model_to_use}")
            model_to_use = self.valves.t2v_model

        # 检查是否有可用的模型
        if not model_to_use:
            await self.emit_status(message=f"💥错误: 没有配置可用的 T2V 模型。", done=True)
            yield "Error: No T2V model is configured in the settings."
            return
        # --- 模型选择结束 ---

        try:
            await self.emit_status(message=f"🚀正在提交任务到 SiliconFlow (模型: {model_to_use})")

            # 调用更新后的 get_requestId
            requestId = await self.get_requestId(
                pro=prompt,
                model=model_to_use,  # 使用自动选择的模型
                image_size=self.valves.image_size,
                seed=self.valves.seed,
                negative_prompt=self.valves.negative_prompt,
                image=image  # 传入提取到的图像 (可能已被设为 None)
            )

            # loop_get_video 会处理状态轮询和最终结果
            async for chunk in self.loop_get_video(requestId):
                yield chunk

        except Exception as e:
            error_message = f"💥任务提交失败: {str(e)}"
            await self.emit_status(message=error_message, done=True)
            yield error_message
            return
