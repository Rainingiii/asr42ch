import asyncio
import websockets
import pyaudio
import numpy as np
import json
import time
import wave
import collections
from typing import List, Dict, Any, Set
from threading import Thread
import queue
import aiohttp
import os
import uuid
import functools
import math

# ---------- 从 JSON 文件读取配置 ----------
CONFIG_FILE = "config.json"

FILE_FIELD_NAME = "audioFile"
ID_FIELD_NAME = "hearingId"

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"配置文件 {CONFIG_FILE} 不存在，请创建它。")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

API_KEYS = config.get("API_KEYS", [])
LANG = config.get("LANG", "zh")
CONTINUOUS_DECODING = config.get("CONTINUOUS_DECODING", True)
BASE_URL = config.get("BASE_URL", "ws://speech.xiaoyuzhineng.com:12392")
UPLOAD_URL = config.get("UPLOAD_URL", "14.204.16.40:19877")

# SAMPLE_RATE 既用于打开设备流的参考速率，也作为发送/保存前的目标采样率（例如 16000）
SAMPLE_RATE = int(config.get("SAMPLE_RATE", 16000))
FRAMES_PER_BUFFER = int(config.get("FRAMES_PER_BUFFER", 882))

# 通道选择平滑：滑动窗口长度与切换阈值
# 当最近 SELECTION_SMOOTHING_WINDOW 个块中，某通道成为“瞬时最响”次数比例超过阈值时，才切换到该通道
SELECTION_SMOOTHING_WINDOW = int(config.get("SELECTION_SMOOTHING_WINDOW", 20))
SELECTION_SWITCH_THRESHOLD = float(config.get("SELECTION_SWITCH_THRESHOLD", 0.6))

HOST_API_INDEX = config.get("HOST_API_INDEX", 0)
DEVICE_NAMES = config.get("DEVICE_NAMES", [])

WS_HOST = config.get("WS_HOST", "0.0.0.0")
WS_PORT = config.get("WS_PORT", 8081)

# 固定 ring buffer 长度（用于可视化/统计的短历史）
RING_BUFFER_MAXLEN = 0

# 当所有通道的最高 RMS 都低于该阈值时视为静音（不发送）
MIN_ACTIVE_RMS = float(config.get("MIN_ACTIVE_RMS", 0.0005))

# 全局状态
desired_ch = 0
ring_buffers = []  # will be assigned in main()
cache_buffers = []  # per-channel cache for flush-on-switch
selected_once = []  # per-channel flag: whether this channel has been selected at least once
results_storage: List[List[Dict[str, Any]]] = []
connected_clients: Set = set()
connected_clients_lock = None  # 在 main() 中初始化为 asyncio.Lock()
recognition_active = False
recording_paused = False
pause_start_time = None
total_paused_duration = 0.0

# 只保存被选中通道的单声道音频
selected_channel_queue: asyncio.Queue = None
recording_task: asyncio.Task = None
recording_filename = None
start_time = None
current_session_id = None
current_hotwords: List[str] = []

# 管理 channel worker 与 ws 连接的全局变量
ws_connections = []
channel_tasks: List[asyncio.Task] = []
broadcast_queue: asyncio.Queue = None

# ---------------- helper: list host apis & devices ----------------
def list_host_apis_and_devices(p: pyaudio.PyAudio):
    print("=== Host APIs ===")
    try:
        for i in range(p.get_host_api_count()):
            info = p.get_host_api_info_by_index(i)
            print(f"hostApi index={i}, name={info.get('name')}")
    except Exception as e:
        print("列出 host apis 失败:", e)
    print("=== Devices ===")
    try:
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            print(
                f"dev index={i}, name={info.get('name')}, hostApi={info.get('hostApi')}, "
                f"maxInputChannels={info.get('maxInputChannels')}, defaultRate={info.get('defaultSampleRate')}"
            )
    except Exception as e:
        print("列出 devices 失败:", e)
    print("=================")

# ---------- 重采样 ----------
def resample_matrix_linear(frames_matrix: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    线性插值重采样：frames_matrix shape = (frames, channels)
    """
    if orig_sr == target_sr:
        return frames_matrix
    frame_count = frames_matrix.shape[0]
    if frame_count <= 1:
        new_len = int(round(frame_count / orig_sr * target_sr))
        return np.tile(frames_matrix, (new_len, 1))[:new_len, :]
    duration = frame_count / orig_sr
    new_len = int(round(duration * target_sr))
    orig_idx = np.linspace(0, frame_count - 1, frame_count)
    new_idx = np.linspace(0, frame_count - 1, new_len)
    channels = frames_matrix.shape[1]
    resampled = np.zeros((new_len, channels), dtype=frames_matrix.dtype)
    for c in range(channels):
        resampled[:, c] = np.interp(new_idx, orig_idx, frames_matrix[:, c])
    return resampled

# ---------- WAV 保存（只保存被选中通道，异步写入） ----------
async def save_selected_channel_audio(selected_queue: asyncio.Queue, filename: str, rate: int):
    """
    selected_queue 中传入的是连续的 mono int16 bytes（采样率为 rate），直到收到 None 表示结束并关闭文件。
    """
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(rate)
    print(f"[save-selected] 开始保存单声道（被选中通道）音频到 {filename} (rate={rate})")
    try:
        while True:
            data = await selected_queue.get()
            if data is None:
                break
            if not data:
                continue
            try:
                wf.writeframes(data)
            except Exception as e:
                print("[save-selected] writeframes error:", e)
                continue
    except Exception as e:
        print("[save-selected] 保存主循环异常:", e)
    finally:
        try:
            wf.close()
        except Exception:
            pass
        print(f"[save-selected] 已保存并关闭：{filename}")

# ---------- WebSocket server（控制 start/stop/pause/resume/save, 增加 update_hotwords） ----------
async def ws_server_handler(ws):
    peer = None
    global recognition_active, recording_paused, pause_start_time, total_paused_duration, current_hotwords
    try:
        try:
            peer = ws.remote_address
        except Exception:
            peer = None
        print(f"[ws server] 客户端连接: {peer}")

        try:
            async with connected_clients_lock:
                connected_clients.add(ws)
        except Exception:
            pass

        try:
            await ws.send(json.dumps({"type": "status", "msg": "connected"}))
        except Exception as e:
            print(f"[ws server] 发送连接成功消息失败: {e}")

        try:
            status_payload = {
                "type": "status",
                "msg": "start_recording",
                "recognition_active": bool(recognition_active),
                "recording_paused": bool(recording_paused),
                "session_id": current_session_id if current_session_id is not None else "",
                "recording_filename": recording_filename if recording_filename is not None else "",
                "start_time": start_time if start_time is not None else None,
                "hotwords": current_hotwords.copy() if current_hotwords else []
            }
            try:
                await ws.send(json.dumps(status_payload, ensure_ascii=False))
            except Exception:
                pass
        except Exception:
            pass

        async for message in ws:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                action = data.get("action")
                session_id = data.get("session_id")
            except Exception:
                continue

            if msg_type == "cmd":
                if action == "start_recording":
                    if not session_id:
                        await safe_send(ws, {"type": "cmd_error", "msg": "缺少 session_id，无法开始录音"})
                        continue
                    await start_recognition(session_id)
                    await safe_send(ws, {
                        "type": "cmd",
                        "action": "start_recording_confirmed",
                        "session_id": session_id,
                        "msg": f"start_recording_confirmed! 保存为文件 {current_session_id}.wav"
                    })

                elif action == "stop_recording":
                    await stop_recognition()
                    await safe_send(ws, {
                        "type": "cmd",
                        "action": "stop_recording",
                        "session_id": session_id,
                        "msg": f"stop_recording_confirmed! 文件 {recording_filename} 已保存"
                    })

                elif action == "pause_recording":
                    if not recognition_active:
                        await safe_send(ws, {"type":"cmd_error","action":"pause_recording","msg":"当前没有正在进行的识别，无法暂停"})
                        continue
                    if recording_paused:
                        await safe_send(ws, {"type":"cmd","action":"pause_recording","msg":"已经处于暂停状态"})
                        continue
                    recording_paused = True
                    pause_start_time = time.time()
                    print(f"[ws server] 已暂停录音/识别 (session_id={session_id})")
                    await safe_send(ws, {"type":"cmd","action":"pause_recording_confirmed","session_id":session_id,"msg":"已暂停写入音频与识别"})

                elif action == "resume_recording":
                    if not recognition_active:
                        await safe_send(ws, {"type":"cmd_error","action":"resume_recording","msg":"当前没有正在进行的识别，无法恢复"} )
                        continue
                    if not recording_paused:
                        await safe_send(ws, {"type":"cmd","action":"resume_recording","msg":"当前没有处于暂停状态"} )
                        continue
                    if pause_start_time is not None:
                        elapsed = time.time() - pause_start_time
                        total_paused_duration += elapsed
                        pause_start_time = None
                    recording_paused = False
                    print(f"[ws server] 已恢复录音/识别 (session_id={session_id})")
                    await safe_send(ws, {"type":"cmd","action":"resume_recording_confirmed","session_id":session_id,"msg":"已恢复写入音频与识别"})

                elif action == "save_recording":
                    url = data.get("url")
                    if url:
                        url = url.replace("/api", "")
                        url = f"{UPLOAD_URL}{url}"
                    the_id = data.get("id")
                    file_path = f"{session_id}.wav"

                    if not session_id or not url or not the_id or not os.path.exists(file_path):
                        await safe_send(ws, {"type":"cmd_error","action":"save_recording","session_id":session_id,"msg":"参数缺失或文件不存在"} )
                        continue

                    await safe_send(ws, {
                        "type":"cmd",
                        "action":"save_recording_started",
                        "session_id":session_id,
                        "msg": f"开始后台上传 {file_path} 到 {url}"
                    })

                    asyncio.create_task(background_upload(file_path, url, the_id, session_id))

                elif action == "update_hotwords":
                    try:
                        values = data.get("values", [])
                        if values is None:
                            values = []
                        values = [str(v) for v in values if v is not None]
                    except Exception:
                        values = []

                    current_hotwords = values.copy()

                    if len(values) == 0:
                        hotword_text = "实时热词:"
                    else:
                        hotword_text = "实时热词:" + ",".join(values)

                    print(f"[ws server] 收到 update_hotwords, 将发送到各通道: {hotword_text}")

                    send_tasks = []
                    sent_channels = []
                    for i, w in enumerate(list(ws_connections) if isinstance(ws_connections, list) else []):
                        if w is None:
                            continue
                        try:
                            closed = getattr(w, "closed", False)
                        except Exception:
                            closed = False
                        if closed:
                            try:
                                ws_connections[i] = None
                            except Exception:
                                pass
                            continue

                        async def _send_to_channel(idx, ws_conn):
                            try:
                                await ws_conn.send(hotword_text)
                                return idx, True, None
                            except Exception as e_send:
                                return idx, False, str(e_send)

                        send_tasks.append(_send_to_channel(i, w))

                    if send_tasks:
                        results = await asyncio.gather(*send_tasks, return_exceptions=False)
                        for idx, ok, err in results:
                            if ok:
                                sent_channels.append(idx)
                            else:
                                print(f"[hotwords] 发送到通道 {idx} 失败: {err}")
                                try:
                                    await ws_connections[idx].close()
                                except Exception:
                                    pass
                                try:
                                    ws_connections[idx] = None
                                except Exception:
                                    pass

                    await safe_send(ws, {
                        "type": "cmd",
                        "action": "update_hotwords_confirmed",
                        "session_id": session_id,
                        "msg": f"热词已发送: {hotword_text}",
                        "sent_channels": sent_channels
                    })

    except Exception as e:
        print(f"[ws server] handler 异常: {e}")
    finally:
        try:
            async with connected_clients_lock:
                connected_clients.discard(ws)
        except Exception:
            pass
        try:
            print(f"[ws server] 客户端断开: {peer}")
        except Exception:
            pass

# ---------- 辅助函数：安全发送 WebSocket 消息 ----------
async def safe_send(ws, data):
    try:
        await ws.send(json.dumps(data, ensure_ascii=False))
    except Exception as e:
        print(f"[ws server] ws.send 异常 (客户端可能已关闭): {e}")

# ---------- 后台上传任务 ----------
async def background_upload(file_path, url, the_id, session_id):
    print(f"[ws server] 后台上传开始: {file_path} -> {url}")
    try:
        ok, status, resp_text = await upload_file_multipart(url, file_path, the_id, timeout_seconds=60)
        short_resp = (resp_text[:1000] + '...') if resp_text and len(resp_text) > 1000 else resp_text
        if ok:
            print(f"[ws server] 上传成功: {file_path} -> {url} (HTTP {status})")
        else:
            print(f"[ws server] 上传失败: {file_path} -> {url} (HTTP {status}), {short_resp}")
    except Exception as e:
        print(f"[ws server] 上传异常: {file_path} -> {url}, {e}")

# ---------- 异步上传文件 ----------
async def upload_file_multipart(url: str, file_path: str, id_value: str,
                                timeout_seconds: int = 30, headers: dict = None):
    if not os.path.exists(file_path):
        return False, None, f"文件不存在: {file_path}"
    f = None
    try:
        f = open(file_path, "rb")
        data = aiohttp.FormData()
        data.add_field(ID_FIELD_NAME, str(id_value))
        data.add_field(FILE_FIELD_NAME, f,
                       filename=os.path.basename(file_path),
                       content_type="audio/wav")

        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as sess:
            async with sess.post(url, data=data) as resp:
                try:
                    text = await resp.text()
                except Exception:
                    raw = await resp.read()
                    text = raw.decode(errors="ignore")
                ok = 200 <= resp.status < 300
                return ok, resp.status, text
    except Exception as e:
        return False, None, f"上传异常: {e}"
    finally:
        try:
            if f:
                f.close()
        except Exception:
            pass

# ---------- broadcaster（广播识别结果/状态到已连接客户端） ----------
async def broadcaster(broadcast_queue: asyncio.Queue):
    while True:
        item = await broadcast_queue.get()
        if item is None:
            break
        text = json.dumps(item, ensure_ascii=False)
        coros = []
        to_remove = []

        try:
            async with connected_clients_lock:
                clients = list(connected_clients)
        except Exception:
            clients = list(connected_clients)

        for ws in clients:
            try:
                closed = getattr(ws, "closed", False)
            except Exception:
                closed = False
            if closed:
                to_remove.append(ws)
                continue
            coros.append(_safe_send(ws, text, to_remove))

        if coros:
            await asyncio.gather(*coros, return_exceptions=True)

        if to_remove:
            try:
                async with connected_clients_lock:
                    for ws in to_remove:
                        connected_clients.discard(ws)
            except Exception:
                pass

async def _safe_send(ws, text: str, to_remove: List):
    try:
        await ws.send(text)
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
        to_remove.append(ws)

# ---------- 控制识别 ----------
async def start_recognition(session_id: str):
    global recognition_active, recording_task, selected_channel_queue, recording_filename, start_time
    global recording_paused, pause_start_time, total_paused_duration, current_session_id
    global channel_tasks, ws_connections, broadcast_queue, desired_ch, selected_once, cache_buffers

    if recognition_active:
        print("[ws server] 已有识别在进行中，忽略 start_recording")
        return

    current_session_id = f"{session_id}"

    recognition_active = True
    start_time = time.time()

    recording_paused = False
    pause_start_time = None
    total_paused_duration = 0.0

    # 改为包含 uuid 的文件名，避免覆盖
    recording_filename = f"{current_session_id}.wav"

    # 创建 selected_channel_queue 并启动写入任务（只保存被选中通道）
    selected_channel_queue = asyncio.Queue()
    # 确保我们以 SAMPLE_RATE 写入 WAV（文件头为 SAMPLE_RATE）
    recording_task = asyncio.create_task(
        save_selected_channel_audio(selected_channel_queue, recording_filename, SAMPLE_RATE)
    )
    print(f"[ws server] 开始识别 文件: {recording_filename} (internal_session={current_session_id})")

    if broadcast_queue is None:
        broadcast_queue = asyncio.Queue()

    if not isinstance(ws_connections, list) or len(ws_connections) < desired_ch:
        ws_connections = [None for _ in range(desired_ch)]

    # 重置 selected_once（start 时确保为未选中状态）
    try:
        selected_once = [False for _ in range(desired_ch)]
    except Exception:
        selected_once = [False] * desired_ch

    # 清理旧 channel_tasks（如果存在）
    try:
        if isinstance(channel_tasks, list) and channel_tasks:
            for t in channel_tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await asyncio.gather(*channel_tasks, return_exceptions=True)
            channel_tasks.clear()
    except Exception as e:
        print("[start_recognition] 清理旧 channel_tasks 异常:", e)

    # 启动 channel workers（仅在开始识别时启动）
    try:
        for ch in range(desired_ch):
            api_key = API_KEYS[ch] if ch < len(API_KEYS) else (API_KEYS[0] if API_KEYS else "")
            t = asyncio.create_task(channel_worker_direct(ch, api_key, broadcast_queue))
            channel_tasks.append(t)
        print(f"[start_recognition] 已为 {desired_ch} 个通道启动 channel_workers")
    except Exception as e:
        print("[start_recognition] 启动 channel workers 异常:", e)

async def stop_recognition():
    global recognition_active, recording_task, selected_channel_queue, start_time, recording_filename
    global recording_paused, pause_start_time, total_paused_duration
    global results_storage, ring_buffers, cache_buffers, current_session_id, ws_connections, channel_tasks, selected_once

    if not recognition_active:
        print("[ws server] 没有进行中的识别，忽略 stop_recording")
        return

    print("[stop_recognition] 开始停止识别流程...")
    recognition_active = False

    if recording_paused and pause_start_time is not None:
        total_paused_duration += (time.time() - pause_start_time)
        pause_start_time = None
        recording_paused = False

    end_time = round(time.time(), 2)

    # 先通知保存队列结束
    if selected_channel_queue is not None:
        try:
            await selected_channel_queue.put(None)
        except Exception as e_put:
            print("[stop_recognition] selected_channel_queue.put(None) 异常:", e_put)
    if recording_task is not None:
        try:
            await recording_task
        except Exception as e_rec:
            print("[stop_recognition] waiting recording_task 异常:", e_rec)

    # 关闭 ASR ws connections
    try:
        if isinstance(ws_connections, list):
            for i, ws in enumerate(ws_connections):
                if ws is not None:
                    try:
                        await ws.close()
                        print(f"[stop_recognition] 已关闭 ws_connections[{i}]")
                    except Exception as e_close:
                        print(f"[stop_recognition] 关闭 ws_connections[{i}] 异常: {e_close}")
                    finally:
                        try:
                            ws_connections[i] = None
                        except Exception:
                            pass
        print("[stop_recognition] 已尝试关闭所有 ws_connections")
    except Exception as e:
        print("[stop_recognition] 关闭 ws_connections 总体异常:", e)

    # 取消 channel tasks
    try:
        if isinstance(channel_tasks, list) and channel_tasks:
            for t in channel_tasks:
                try:
                    t.cancel()
                except Exception:
                    pass
            await asyncio.gather(*channel_tasks, return_exceptions=True)
            channel_tasks.clear()
            print("[stop_recognition] channel_tasks 已取消并清理")
    except Exception as e:
        print("[stop_recognition] 取消 channel_tasks 异常:", e)

    # 清空本地结果与缓冲
    try:
        if isinstance(results_storage, list):
            for ch_list in results_storage:
                try:
                    ch_list.clear()
                except Exception:
                    pass
        for dq in ring_buffers:
            try:
                dq.clear()
            except Exception:
                pass
        for dq in cache_buffers:
            try:
                dq.clear()
            except Exception:
                pass
        # reset selected_once
        try:
            selected_once = [False for _ in range(len(selected_once))]
        except Exception:
            selected_once = []
        print("[stop_recognition] 已清空 results_storage 与 ring_buffers 与 cache_buffers 并重置 selected_once")
    except Exception as e:
        print("[stop_recognition] 清空本地识别结果/缓冲时异常:", e)

    try:
        current_session_id = None
    except Exception:
        pass

    duration = end_time - start_time - total_paused_duration if start_time is not None else 0.0
    if duration < 0:
        duration = 0.0
    print(f"[ws server] 停止识别 文件: {recording_filename} 时长: {duration:.2f} 秒")

# ---------- 通道 WebSocket worker（识别结果转发，支持自动重连与提示，重连后下发热词） ----------
async def channel_worker_direct(chan_id: int, api_key: str, broadcast_queue: asyncio.Queue):
    max_backoff = 30
    backoff = 1

    while True:
        ts = str(int(time.time()))
        url = f"{BASE_URL}?apikey={api_key}&ts={ts}&lang={LANG}&continuous_decoding={str(CONTINUOUS_DECODING).lower()}"

        try:
            async with websockets.connect(url) as ws:
                try:
                    old_conn = None
                    if isinstance(ws_connections, list) and chan_id < len(ws_connections):
                        old_conn = ws_connections[chan_id]
                    ws_connections[chan_id] = ws
                except Exception:
                    pass

                print(f"[chan {chan_id}] WebSocket 已连接")

                try:
                    if broadcast_queue is not None:
                        await broadcast_queue.put({
                            "type": "status",
                            "chan": chan_id,
                            "status": "asr_connected",
                            "msg": f"chan {chan_id} -> ASR 已连接"
                        })
                except Exception:
                    pass

                backoff = 1

                try:
                    if current_hotwords:
                        hotword_text = "实时热词:" + ",".join(current_hotwords)
                    else:
                        hotword_text = "实时热词:"
                    try:
                        await ws.send(hotword_text)
                    except Exception:
                        pass
                except Exception:
                    pass

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        action = msg.get("action")

                        if not recognition_active or current_session_id is None:
                            continue

                        if action in ("partial_result", "final_result"):
                            nbest = msg.get("nbest", [])
                            if not nbest:
                                continue
                            pieces = nbest[0].get("pieces", [])
                            words = [p.get("word", "") for p in pieces]
                            transcript = "".join(words)

                            if start_time is not None:
                                ts_now = time.time() - start_time
                                ts_now = round(ts_now, 2)
                            else:
                                ts_now = None

                            if transcript.strip() == "嗯":
                                continue

                            print(f"[chan {chan_id}] {action}:识别: {transcript} (time={ts_now}s)")
                            try:
                                await broadcast_queue.put({
                                    "type": "content",
                                    "chan": chan_id,
                                    "time": ts_now,
                                    "action": action,
                                    "text": transcript
                                })
                            except Exception:
                                pass
                    except Exception:
                        continue

        except asyncio.CancelledError:
            print(f"[chan {chan_id}] channel task cancelled，退出重连循环")
            try:
                if isinstance(ws_connections, list) and chan_id < len(ws_connections):
                    conn = ws_connections[chan_id]
                    try:
                        if conn is not None:
                            try:
                                await conn.close()
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        if ws_connections[chan_id] is conn:
                            ws_connections[chan_id] = None
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                if broadcast_queue is not None:
                    await broadcast_queue.put({
                        "type": "status",
                        "chan": chan_id,
                        "status": "asr_disconnected",
                        "msg": f"chan {chan_id} 已取消并退出"
                    })
            except Exception:
                pass

            break

        except Exception as e:
            print(f"[chan {chan_id}] websocket 异常: {e}")
            try:
                if isinstance(ws_connections, list) and chan_id < len(ws_connections):
                    try:
                        conn = ws_connections[chan_id]
                        if conn is not None:
                            try:
                                await conn.close()
                            except Exception:
                                pass
                        if ws_connections[chan_id] is conn:
                            ws_connections[chan_id] = None
                    except Exception:
                        pass
            except Exception:
                pass

            try:
                if broadcast_queue is not None:
                    await broadcast_queue.put({
                        "type": "status",
                        "chan": chan_id,
                        "status": "asr_disconnected",
                        "msg": f"chan {chan_id} 与 ASR 断开，{backoff}s 后重连: {e}"
                    })
            except Exception:
                pass

            try:
                await asyncio.sleep(backoff)
            except asyncio.CancelledError:
                break
            backoff = min(max_backoff, backoff * 2)

    print(f"[chan {chan_id}] worker 结束。")

# ---------- 从多个 stereo-pair 设备启动阻塞读线程 ----------
def start_stream_reader_threads(p: pyaudio.PyAudio, device_indices: List[int], rate: int, frames_per_buffer: int):
    """
    返回 streams, device_queues, channel_counts, device_rates
    device_rates 是每个设备真实用于打开流的采样率（int）
    """
    streams = []
    device_queues = []
    channel_counts = []
    device_rates = []
    try:
        for di in device_indices:
            info = p.get_device_info_by_index(di)
            ch = int(info.get('maxInputChannels', 2) or 2)
            # 尝试使用设备的 defaultSampleRate 打开流，避免驱动做隐式重采样
            try:
                default_rate = info.get('defaultSampleRate')
                dev_rate = int(default_rate) if default_rate is not None else int(rate)
            except Exception:
                dev_rate = int(rate)
            try:
                stream = p.open(format=pyaudio.paInt16, channels=ch, rate=int(dev_rate),
                                input=True, input_device_index=di,
                                frames_per_buffer=frames_per_buffer, stream_callback=None)
            except Exception as e_open:
                for s, _ in streams:
                    try:
                        if s.is_active():
                            s.stop_stream()
                    except Exception:
                        pass
                    try:
                        s.close()
                    except Exception:
                        pass
                raise RuntimeError(f"打开设备 idx={di} (name={info.get('name')}) 失败: {e_open}") from e_open

            q = queue.Queue(maxsize=200)

            def reader_loop(s, q_local, frames_per_buffer_local):
                while True:
                    try:
                        data = s.read(frames_per_buffer_local, exception_on_overflow=False)
                    except Exception:
                        break
                    try:
                        q_local.put(data, timeout=1)
                    except Exception:
                        pass

            t = Thread(target=reader_loop, args=(stream, q, frames_per_buffer), daemon=True)
            t.start()
            streams.append((stream, t))
            device_queues.append(q)
            channel_counts.append(ch)
            device_rates.append(dev_rate)
    except Exception:
        raise
    return streams, device_queues, channel_counts, device_rates

# ---------- 重型同步处理函数（放到线程池中执行） ----------
def _sync_process_parts(parts, channel_counts, desired_ch_local, device_rates, target_sr):
    """
    parts: bytes 列表（每个元素来自一个设备）
    channel_counts: 每个设备的通道数
    device_rates: 每个设备的实际采样率（Hz）
    target_sr: 目标采样率（SAMPLE_RATE）
    返回:
      - pcm_bytes: interleaved int16 bytes（以 target_sr 为时间基准）
      - per_channel_frames: list(len=desired_ch) 每项为该通道在 target_sr 下的连续 int16 bytes（大窗口）
    """
    import numpy as np

    mats = []
    frame_counts = []
    for i, data in enumerate(parts):
        if not data:
            mats.append(None)
            frame_counts.append(0)
            continue
        try:
            arr = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        except Exception:
            mats.append(None)
            frame_counts.append(0)
            continue

        ch = channel_counts[i]
        if ch <= 0:
            mats.append(None)
            frame_counts.append(0)
            continue
        fc = int(arr.size // ch)
        if fc == 0:
            mats.append(None)
            frame_counts.append(0)
            continue
        arr = arr.reshape((fc, ch))  # frames x channels, 以设备的 dev_rate 为时间基准

        # 先把每个设备的数据重采样到目标采样率 target_sr（如果需要）
        try:
            dev_rate = int(device_rates[i])
        except Exception:
            dev_rate = int(target_sr)

        # **只在设备采样率与目标不同的情况下进行重采样**
        if dev_rate != int(target_sr):
            try:
                arr = resample_matrix_linear(arr, dev_rate, int(target_sr))
            except Exception:
                # 若重采样失败，则退回原 arr（但注意时间轴可能不一致）
                pass

        mats.append(arr)
        frame_counts.append(arr.shape[0])

    min_fc_candidates = [fc for fc in frame_counts if fc > 0]
    if not min_fc_candidates:
        return b'', [b'' for _ in range(desired_ch_local)]

    # 对齐：取最小帧数（现在 mats 已在 target_sr 下）
    min_fc = min(min_fc_candidates)

    pieces = []
    for m in mats:
        if m is None:
            pieces.append(np.zeros((min_fc, 1), dtype=np.float32))
        else:
            pieces.append(m[:min_fc, :])
    frames_matrix = np.concatenate(pieces, axis=1)
    frames_matrix = frames_matrix[:, :desired_ch_local]

    if frames_matrix.size == 0:
        return b'', [b'' for _ in range(desired_ch_local)]

    interleaved_pcm16 = np.clip(frames_matrix, -1.0, 1.0)
    interleaved_pcm16 = (interleaved_pcm16 * 32767).astype(np.int16)
    pcm_bytes = interleaved_pcm16.tobytes()

    per_channel_frames = [b'' for _ in range(desired_ch_local)]
    for ch in range(desired_ch_local):
        col = interleaved_pcm16[:, ch]
        per_channel_frames[ch] = col.tobytes()

    return pcm_bytes, per_channel_frames

# ---------- 合并并处理来自多个 device 的帧（已将重型计算移入线程池） ----------
async def combined_reader_loop(device_queues: List[queue.Queue], channel_counts: List[int],
                               broadcast_queue: asyncio.Queue, loop, device_rates: List[int]):
    global recognition_active, selected_channel_queue, desired_ch, recording_paused, ring_buffers, cache_buffers, ws_connections, selected_once
    total_channels = sum(channel_counts)
    print(f"[combined_reader] total_channels = {total_channels}, channel_counts = {channel_counts}")
    # 如果 device_rates 中的某些设备采样率等于 SAMPLE_RATE，则对于这些设备不会进行重采样
    # 滑动窗口平滑通道选择：维护最近若干块的“瞬时赢家”，仅在出现比例超过阈值时切换
    winners_window = collections.deque(maxlen=SELECTION_SMOOTHING_WINDOW if SELECTION_SMOOTHING_WINDOW > 0 else 1)
    current_selected_ch = 0

    while True:
        try:
            gets = [loop.run_in_executor(None, q.get) for q in device_queues]
            parts = await asyncio.gather(*gets)
        except Exception as e:
            print("[combined_reader] 从 device_queues 读取异常:", e)
            break

        try:
            func = functools.partial(_sync_process_parts, parts, channel_counts, desired_ch, device_rates, SAMPLE_RATE)
            pcm_bytes, per_channel_frames = await loop.run_in_executor(None, func)
        except Exception as e:
            print("[combined_reader] 处理帧异常:", e)
            await asyncio.sleep(0.001)
            continue

        if not pcm_bytes:
            await asyncio.sleep(0)
            continue

        # ----------------------------
        # 选能量最大的通道用于“保存”和“发送”（持续送流，不做静音阈值门控）
        # - 保存：始终将被选中通道写入（除非处于暂停），确保文件时长与真实时长一致
        # - 发送：仅将被选中通道持续发送到对应 ASR 连接（包含静音片段）
        # per_channel_frames 中的 bytes 已是目标采样率 SAMPLE_RATE（例如 16k）
        # ----------------------------
        try:
            # 若处于暂停，既不保存也不发送
            if recording_paused:
                await asyncio.sleep(0)
                continue

            # 计算每通道 RMS（用于瞬时赢家判断）
            rms_list = []
            for ch in range(min(desired_ch, len(per_channel_frames))):
                chunk = per_channel_frames[ch]
                if not chunk:
                    rms_list.append(0.0)
                    continue
                try:
                    arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
                    if arr.size == 0:
                        rms = 0.0
                    else:
                        rms = float(np.sqrt(np.mean(arr * arr)))
                    rms_list.append(rms)
                except Exception:
                    rms_list.append(0.0)

            if rms_list:
                selected_ch_instant = int(np.argmax(np.array(rms_list)))  # 瞬时赢家
            else:
                selected_ch_instant = 0  # 无有效信号时退回 0

            # --- 滑动窗口平滑逻辑 ---
            previous_selected = current_selected_ch
            try:
                winners_window.append(selected_ch_instant)
                # 只有当瞬时赢家在窗口内的占比超过阈值时，才切换到该通道
                window_len = len(winners_window)
                if window_len > 0:
                    count_winner = winners_window.count(selected_ch_instant)
                    ratio = count_winner / window_len
                    if ratio >= SELECTION_SWITCH_THRESHOLD:
                        current_selected_ch = selected_ch_instant
                # 使用平滑后的通道作为本次发送/保存通道
                selected_ch = current_selected_ch
            except Exception:
                # 任何异常下退回即时选择（保持鲁棒性）
                selected_ch = selected_ch_instant
                current_selected_ch = selected_ch_instant

            # ---------- 检测到真正的切换：处理新选中通道的缓存 ----------
            try:
                if current_selected_ch != previous_selected:
                    idx = current_selected_ch
                    if isinstance(cache_buffers, list) and 0 <= idx < len(cache_buffers):
                        # 如果是第一次被选中：**不 flush 不写入**，直接清空缓存并标记已选中
                        if not (isinstance(selected_once, list) and idx < len(selected_once) and selected_once[idx]):
                            # first time select: skip flushing and writing to selected_channel_queue
                            try:
                                selected_once[idx] = True
                            except Exception:
                                # ensure list length
                                try:
                                    selected_once = [False for _ in range(desired_ch)]
                                    selected_once[idx] = True
                                except Exception:
                                    pass
                            try:
                                cache_buffers[idx].clear()
                            except Exception:
                                pass
                            print(f"[combined_reader] 通道 {idx} 第一次被选中，已清空其缓存（不发送历史帧以避免重复）")
                        else:
                            # 非第一次：按旧逻辑 flush 缓存到保存队列与 ASR
                            cached_items = list(cache_buffers[idx])  # snapshot（不含本循环的当前帧）
                            if cached_items:
                                print(f"[combined_reader] 检测到切换：{previous_selected} -> {current_selected_ch}，开始 flush 通道 {idx} 的缓存 (len={len(cached_items)})")
                                try:
                                    ws_conn = None
                                    if isinstance(ws_connections, list) and idx < len(ws_connections):
                                        ws_conn = ws_connections[idx]
                                except Exception:
                                    ws_conn = None

                                # 逐帧按时间顺序发送：先写入保存队列（保持文件完整），再尝试发送到 ASR（如果存在）
                                for chunk in cached_items:
                                    if not chunk:
                                        continue
                                    # 写入保存队列（非阻塞）
                                    try:
                                        try:
                                            asyncio.run_coroutine_threadsafe(selected_channel_queue.put(chunk), loop)
                                        except Exception:
                                            try:
                                                await selected_channel_queue.put(chunk)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                    # 发送到 ASR（若有连接）
                                    if ws_conn is not None:
                                        try:
                                            closed = getattr(ws_conn, "closed", False)
                                        except Exception:
                                            closed = False
                                        if not closed:
                                            try:
                                                await ws_conn.send(chunk)
                                            except Exception:
                                                pass
                                # 清空缓存（已 flush）
                                try:
                                    cache_buffers[idx].clear()
                                except Exception:
                                    pass
                                print(f"[combined_reader] flush 完成: 通道 {idx}")
            except Exception as e:
                print("[combined_reader] flush 缓存异常:", e)

            # 1) 始终将被选中通道写入保存队列（保持文件时间轴完整）
            chunk_to_save = per_channel_frames[selected_ch] if selected_ch < len(per_channel_frames) else b''
            if selected_channel_queue is not None and chunk_to_save:
                try:
                    try:
                        asyncio.run_coroutine_threadsafe(selected_channel_queue.put(chunk_to_save), loop)
                    except Exception:
                        try:
                            await selected_channel_queue.put(chunk_to_save)
                        except Exception:
                            pass
                except Exception:
                    pass

            # 2) 仅将“能量最大”的通道持续发送到识别端，并记录到 ring_buffers（用于可视化/短历史）以及追加到 cache_buffers（为将来可能的切换保存历史）
            try:
                chunk_to_send = per_channel_frames[selected_ch]
            except Exception:
                chunk_to_send = b''

            if chunk_to_send:
                # 记录短历史（若启用）
                try:
                    if isinstance(ring_buffers, list) and selected_ch < len(ring_buffers) and ring_buffers[selected_ch] is not None:
                        ring_buffers[selected_ch].append(chunk_to_send)
                except Exception:
                    pass

                # 发送到对应通道的 ASR 连接（非阻塞）
                try:
                    ws_conn = None
                    if isinstance(ws_connections, list) and selected_ch < len(ws_connections):
                        ws_conn = ws_connections[selected_ch]
                    if ws_conn is not None:
                        try:
                            asyncio.run_coroutine_threadsafe(ws_conn.send(chunk_to_send), loop)
                        except Exception:
                            pass
                except Exception:
                    pass

            # 最后：将每个通道本次的帧追加到对应 cache_buffers（为未来可能的切换保存历史）
            try:
                for ch in range(min(desired_ch, len(per_channel_frames))):
                    c = per_channel_frames[ch]
                    if not c:
                        continue
                    try:
                        cache_buffers[ch].append(c)
                    except Exception:
                        pass
            except Exception:
                pass

        except Exception as e:
            print("[combined_reader] 选通道/发送/保存 处理异常:", e)

# ---------- 主流程 ----------
async def main():
    global results_storage, desired_ch, ws_connections, channel_tasks, broadcast_queue, connected_clients_lock
    global ring_buffers, cache_buffers, selected_once
    p = pyaudio.PyAudio()
    server = None
    try:
        # list_host_apis_and_devices(p)

        device_indices = []
        for name in DEVICE_NAMES:
            found = -1
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info.get('hostApi') != HOST_API_INDEX:
                    continue
                if name.lower() in str(info.get('name', '')).lower():
                    found = i
                    break
            if found < 0:
                print(f"[ERROR] 未在 hostApi={HOST_API_INDEX} 下找到设备名包含: '{name}'。请确认设备名并修改 DEVICE_NAMES 或 HOST_API_INDEX。")
                return
            device_indices.append(found)

        print(f"[main] 选择到的设备索引 (hostApi={HOST_API_INDEX}): {device_indices}")

        streams_for_multi, device_queues, channel_counts, device_rates = start_stream_reader_threads(
            p, device_indices, SAMPLE_RATE, FRAMES_PER_BUFFER
        )

        total_ch = sum(channel_counts)
        desired_ch = min(total_ch, len(API_KEYS), 8)  # 限制到 8 通道（4x2）
        print(f"[main] channel_counts={channel_counts}, total_ch={total_ch}, desired_ch={desired_ch}")
        print(f"[main] device_rates (per device): {device_rates}")
        # 提示：如果 device_rates 中有 != SAMPLE_RATE 的值，脚本会在发送/保存前把对应设备数据重采样到 SAMPLE_RATE

        # 初始化结构
        results_storage = [[] for _ in range(desired_ch)]
        # 计算每个通道用于 flush 的缓存长度（基于滑动窗口与阈值），至少保留 1 块
        cache_len = max(1, int(math.ceil(SELECTION_SMOOTHING_WINDOW * SELECTION_SWITCH_THRESHOLD)))
        ring_buffers = [collections.deque(maxlen=RING_BUFFER_MAXLEN) for _ in range(desired_ch)]
        cache_buffers = [collections.deque(maxlen=cache_len) for _ in range(desired_ch)]
        selected_once = [False for _ in range(desired_ch)]
        ws_connections = [None for _ in range(desired_ch)]
        broadcast_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        connected_clients_lock = asyncio.Lock()

        # 启动合并读取协程（注意传 device_rates）
        asyncio.create_task(combined_reader_loop(device_queues, channel_counts, broadcast_queue, loop, device_rates))

        broadcaster_task = asyncio.create_task(broadcaster(broadcast_queue))

        server = await websockets.serve(ws_server_handler, WS_HOST, WS_PORT)
        print(f"[ws server] 服务已启动 ws://{WS_HOST}:{WS_PORT}")
        print("已启动 multi-device 采集，等待客户端 start/stop 命令。按 Ctrl+C 停止。")

        await asyncio.Future()

    except RuntimeError as e:
        print("启动时发生错误:", e)
        print("可能原因：设备被占用或当前 hostApi 不支持为每个物理设备单独打开流。")
        print("建议：确认没有其它程序占用设备；或尝试不同的 HOST_API_INDEX；或把设备驱动设置为以单个多通道设备暴露。")
    except KeyboardInterrupt:
        print("收到 KeyboardInterrupt，准备退出...")
    finally:
        try:
            if selected_channel_queue is not None:
                await selected_channel_queue.put(None)
        except Exception:
            pass

        try:
            for t in channel_tasks:
                t.cancel()
            await asyncio.gather(*channel_tasks, return_exceptions=True)
        except Exception:
            pass

        if server is not None:
            server.close()
            await server.wait_closed()

        try:
            await broadcaster_task
        except Exception:
            pass

        for ch in range(desired_ch):
            print(f"[chan {ch}] 总识别条数: {len(results_storage[ch])}")

        try:
            p.terminate()
        except Exception:
            pass

if __name__ == "__main__":
    asyncio.run(main())
