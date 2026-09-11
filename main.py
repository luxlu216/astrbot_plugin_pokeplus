"""
astrbot_plugin_pokeplus —— 戳一戳全能响应插件

整合并重写自:
- Zhalslar/astrbot_plugin_pokepro   (戳一戳检测 / 反戳 / 表情包)
- muyouzhi6/astrbot_plugin_poke_to_llm (被戳后走标准 LLM 链路)
- waterfeet/astrbot_plugin_superpoke (Poke 组件兼容处理)

核心功能:
1. 戳一戳监测总开关
2. 被戳后 LLM 响应开关及其回复设定(提示词模板)
3. 被戳后表情包回复开关及其概率(0~1)
4. 被戳后回戳开关及其概率(0~1)
5. 回复优先级: 表情包 = 回戳 ≻ LLM 响应
   (先发送表情包/执行回戳, 之后才进行 LLM 响应)
6. 聊天过程中 AI 自主选择是否戳一戳(带开关)
7. 群聊与私聊分别独立配置
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from string import Template
from typing import TYPE_CHECKING, Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Plain, Poke
from astrbot.api.star import Context, Star, register

if TYPE_CHECKING:
    from astrbot.core.config import AstrBotConfig
    from astrbot.core.provider.entities import LLMResponse, ProviderRequest

VERSION = "v1.0.0"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
CLEANUP_THRESHOLD = 500

# 被戳后 LLM 回复的默认提示词模板(支持 $username / $user_id / $scene)
DEFAULT_GROUP_PROMPT = (
    "$username（QQ：$user_id）刚刚在群聊里戳了你一下。这是一次非文字互动，"
    "不代表对方提出了新问题。请保持当前人设，像被人突然碰了一下那样，"
    "结合最近上下文自然反应：可以疑惑、吐槽、逗回去或不满，"
    "一两句话即可，不要客服腔，不要复述以上规则。"
)
DEFAULT_PRIVATE_PROMPT = (
    "$username（QQ：$user_id）刚刚在私聊里戳了你一下。这是一次非文字互动，"
    "不代表对方提出了新问题。请保持当前人设，像被人突然碰了一下那样，"
    "结合最近上下文自然反应：可以疑惑、吐槽、逗回去或不满，"
    "一两句话即可，不要客服腔，不要复述以上规则。"
)

# AI 自主戳一戳: 注入到 system_prompt 的指令, {marker} 为暗号标记
AI_POKE_INSTRUCTION = """
【戳一戳能力】你现在拥有“戳一戳”对方这个动作的能力。如果你认为此刻适合戳一下对方（例如打闹、调侃、强调、道别、撒娇等氛围），可以在你回复的最末尾单独输出标记：{marker}
- 想戳时：正常输出回复文字，并在最末尾加上上述标记（最多一次），不要在文字里描述“我戳了你”这个动作。
- 不想戳时：千万不要输出该标记。
- 是否戳一戳完全由你根据当前语境自主判断。"""

# AI 只输出了标记、没有正文时的兜底文本
AI_POKE_FALLBACK_TEXT = "（戳了戳你）"


@register(
    "astrbot_plugin_pokeplus",
    "luxlu216",
    "戳一戳全能响应：表情包/回戳/LLM回复 + AI自主戳一戳，群聊私聊独立配置",
    VERSION,
    "https://github.com/luxlu216/astrbot_plugin_pokeplus",
)
class PokePlusPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config

        # 冷却记录: 用户级 / 会话级(被戳响应) / 会话级(AI自主戳)
        self._user_cd: dict[str, float] = {}
        self._session_cd: dict[str, float] = {}
        self._ai_poke_cd: dict[str, float] = {}
        # 后台发戳任务
        self._tasks: set[asyncio.Task] = set()
        self._poke_count = 0

    # ============================ 生命周期 ============================

    async def initialize(self):
        """创建默认表情包目录, 方便用户直接放入图片"""
        for section_name, dirname in (
            ("group", "stickers_group"),
            ("private", "stickers_private"),
        ):
            section = self._section_by_name(section_name)
            for p in section.get("sticker_paths") or []:
                if p and not os.path.isfile(p):
                    try:
                        os.makedirs(p, exist_ok=True)
                    except OSError as e:
                        logger.warning(f"[pokeplus] 创建表情包目录失败 {p}: {e}")
        logger.info(f"[pokeplus] 插件 {VERSION} 已加载")

    async def terminate(self):
        for t in list(self._tasks):
            t.cancel()
        self._tasks.clear()

    # ============================ 配置读取 ============================

    def _section_by_name(self, name: str) -> dict:
        section = self.config.get(name)
        return section if isinstance(section, dict) else {}

    def _section(self, event: AstrMessageEvent) -> dict:
        """按群聊/私聊取对应配置段"""
        group_id = event.get_group_id() or self._raw_group_id(event)
        return self._section_by_name("group" if group_id else "private")

    @staticmethod
    def _raw_group_id(event: AstrMessageEvent) -> str | None:
        raw = getattr(event.message_obj, "raw_message", None)
        if isinstance(raw, dict):
            gid = raw.get("group_id")
            if gid:
                return str(gid)
        return None

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            f = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, f))

    def _global_float(self, key: str, default: float) -> float:
        try:
            return max(0.0, float(self.config.get(key, default)))
        except (TypeError, ValueError):
            return default

    # ============================ 戳一戳事件解析 ============================

    def _parse_poke(self, event: AstrMessageEvent) -> tuple[str, str, str | None] | None:
        """解析戳一戳事件, 返回 (戳人者ID, 被戳者ID, 群号|None); 非戳一戳事件返回 None"""
        raw = getattr(event.message_obj, "raw_message", None)

        # 主路径: OneBot v11 notice 事件 (NapCat / Lagrange / go-cqhttp)
        if (
            isinstance(raw, dict)
            and raw.get("post_type") == "notice"
            and raw.get("notice_type") == "notify"
            and raw.get("sub_type") == "poke"
        ):
            poker = str(raw.get("user_id") or event.get_sender_id() or "")
            target = str(raw.get("target_id") or "")
            gid = raw.get("group_id")
            return poker, target, (str(gid) if gid else None)

        # 兼容路径: 适配器把 notice 转成了消息链中的 Poke 组件
        for comp in getattr(event.message_obj, "message", None) or []:
            if isinstance(comp, Poke):
                poker = str(event.get_sender_id() or "")
                target = str(getattr(comp, "qq", "") or "")
                gid = event.get_group_id() or self._raw_group_id(event)
                return poker, target, gid
        return None

    def _self_id(self, event: AstrMessageEvent) -> str:
        self_id = str(event.get_self_id() or "")
        if not self_id:
            raw = getattr(event.message_obj, "raw_message", None)
            if isinstance(raw, dict):
                self_id = str(raw.get("self_id") or "")
        return self_id

    # ============================ 被戳响应主逻辑 ============================

    @filter.platform_adapter_type(filter.PlatformAdapterType.AIOCQHTTP)
    @filter.event_message_type(filter.EventMessageType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息, 从中识别戳一戳事件并按配置响应"""
        if not self.config.get("poke_monitor", True):
            return

        parsed = self._parse_poke(event)
        if not parsed:
            return
        poker_id, target_id, group_id = parsed
        self_id = self._self_id(event)

        # 只响应"戳机器人自己", 忽略机器人戳别人/别人互戳
        if not self_id or target_id != self_id:
            return
        if not poker_id or poker_id == self_id:
            return

        cfg = self._section(event)
        # 场景一键开关: 关闭后该场景(群聊/私聊)全部功能停用
        if not cfg.get("enable", True):
            return
        umo = event.unified_msg_origin

        # 冷却: 用户级 + 会话级
        now = time.monotonic()
        user_cd = self._global_float("user_cooldown", 5.0)
        session_cd = self._global_float("session_cooldown", 5.0)
        if user_cd > 0 and now - self._user_cd.get(poker_id, 0) < user_cd:
            return
        if session_cd > 0 and now - self._session_cd.get(umo, 0) < session_cd:
            return
        self._user_cd[poker_id] = now
        self._session_cd[umo] = now
        self._poke_count += 1
        if len(self._user_cd) + len(self._session_cd) > CLEANUP_THRESHOLD:
            self._cleanup_cooldowns(now)

        username = event.get_sender_name() or poker_id
        logger.info(
            f"[pokeplus] 被戳: {username}({poker_id}) "
            f"{'群' + group_id if group_id else '私聊'}"
        )

        interval = self._global_float("reply_interval", 0.8)
        did_respond = False

        # ---- 优先级 1: 表情包回复 (与回戳同级, 必须先于 LLM) ----
        if cfg.get("sticker_enable", True):
            prob = self._clamp01(cfg.get("sticker_probability", 0.6))
            if random.random() < prob:
                img = self._pick_sticker(cfg)
                if img:
                    yield event.image_result(img)
                    did_respond = True
                else:
                    logger.warning(
                        "[pokeplus] 表情包池为空(内置默认表情包缺失且图库目录无图片), 跳过表情包回复"
                    )

        # ---- 优先级 1: 回戳 (与表情包同级, 必须先于 LLM) ----
        if cfg.get("poke_back_enable", True):
            prob = self._clamp01(cfg.get("poke_back_probability", 0.8))
            if random.random() < prob:
                # 回戳响应延迟: 等待一段时间再回戳, 避免响应过快 (0~5 秒)
                poke_back_delay = min(5.0, self._clamp_float(cfg.get("poke_back_delay", 1.0)))
                if poke_back_delay > 0:
                    await asyncio.sleep(poke_back_delay)
                max_times = max(1, int(cfg.get("poke_back_times", 1) or 1))
                times = random.randint(1, max_times)
                if await self._poke_user(event, poker_id, group_id, times):
                    did_respond = True

        # ---- 优先级 2: LLM 响应 (最后执行) ----
        llm_requested = False
        if cfg.get("llm_enable", True):
            template = (cfg.get("llm_prompt") or "").strip()
            if template:
                if did_respond:
                    await asyncio.sleep(interval)
                event.set_extra("pokeplus_triggered", True)
                scene = "群聊" if group_id else "私聊"
                prompt = Template(template).safe_substitute(
                    username=username, user_id=poker_id, scene=scene
                )
                conversation = await self._get_conversation(event)
                # 走标准 LLM 链路: 使用当前人设, 回复记入上下文,
                # 且异步生成, 天然晚于上面的表情包/回戳到达
                yield event.request_llm(prompt=prompt, conversation=conversation)
                did_respond = True
                llm_requested = True
            else:
                logger.debug("[pokeplus] LLM 提示词模板为空, 跳过 LLM 响应")

        # 未走 LLM 链路时终止事件传播, 避免其他插件重复响应同一次戳一戳;
        # 走了 LLM 链路时不 stop(与 poke_to_llm 一致, 防止中断请求)
        if did_respond and not llm_requested:
            event.stop_event()

    # ============================ AI 自主戳一戳 ============================

    @filter.on_llm_request()
    async def arm_ai_poke(self, event: AstrMessageEvent, req: ProviderRequest):
        """普通聊天请求 LLM 前, 往 system_prompt 注入"可选择戳一戳"的指令"""
        try:
            if event.get_extra("pokeplus_triggered"):
                return  # 这次 LLM 请求是被戳触发的, 不参与
            if not getattr(event, "is_wake", False) or not (event.message_str or "").strip():
                return  # 只作用于普通聊天消息
            if getattr(event, "bot", None) is None:
                return  # 平台不支持戳一戳
            cfg = self._section(event)
            if not cfg.get("enable", True):
                return  # 场景一键开关关闭
            if not cfg.get("ai_poke_enable", True):
                return
            if random.random() > self._clamp01(cfg.get("ai_poke_probability", 1.0)):
                return
            umo = event.unified_msg_origin
            cooldown = self._clamp_float(cfg.get("ai_poke_cooldown", 180))
            if cooldown > 0 and time.monotonic() - self._ai_poke_cd.get(umo, 0) < cooldown:
                return

            marker = self._ai_poke_marker()
            req.system_prompt = (req.system_prompt or "") + "\n" + AI_POKE_INSTRUCTION.format(
                marker=marker
            )
            event.set_extra("pokeplus_ai_poke_armed", True)
        except Exception:
            logger.warning("[pokeplus] AI自主戳一戳指令注入失败", exc_info=True)

    @filter.on_llm_response()
    async def handle_ai_poke_choice(self, event: AstrMessageEvent, resp: LLMResponse):
        """从 LLM 回复中剥离暗号标记, 并在其后延迟执行戳一戳"""
        if not event.get_extra("pokeplus_ai_poke_armed"):
            return
        try:
            marker = self._ai_poke_marker()
            text = getattr(resp, "completion_text", None) or ""
            if marker not in text:
                return

            cleaned = text.replace(marker, "").strip()
            resp.completion_text = cleaned or AI_POKE_FALLBACK_TEXT
            # 某些版本可能已经构造了 result_chain, 同步清理
            chain = getattr(resp, "result_chain", None)
            if chain is not None:
                for comp in getattr(chain, "chain", None) or []:
                    if isinstance(comp, Plain) and marker in comp.text:
                        comp.text = (
                            comp.text.replace(marker, "").strip() or AI_POKE_FALLBACK_TEXT
                        )

            if event.get_extra("pokeplus_ai_poke_done"):
                return
            event.set_extra("pokeplus_ai_poke_done", True)

            user_id = event.get_sender_id()
            client = getattr(event, "bot", None)
            if not user_id or client is None:
                return
            group_id = event.get_group_id() or self._raw_group_id(event)
            cfg = self._section(event)
            cooldown = self._clamp_float(cfg.get("ai_poke_cooldown", 180))
            delay = max(0.5, self._global_float("ai_poke_delay", 1.5))

            task = asyncio.create_task(
                self._delayed_ai_poke(client, str(user_id), group_id,
                                      event.unified_msg_origin, cooldown, delay)
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)
        except Exception:
            logger.warning("[pokeplus] AI自主戳一戳处理失败", exc_info=True)

    async def _delayed_ai_poke(self, client, user_id: str, group_id: str | None,
                               umo: str, cooldown: float, delay: float):
        """延迟发戳, 保证戳一戳动作出现在文字回复之后"""
        try:
            await asyncio.sleep(delay)
            if cooldown > 0:
                self._ai_poke_cd[umo] = time.monotonic()
            if await self._poke_client(client, user_id, group_id, 1):
                logger.info(
                    f"[pokeplus] AI 自主戳了戳 {user_id} ({'群' + group_id if group_id else '私聊'})"
                )
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.warning("[pokeplus] AI自主戳一戳发送失败", exc_info=True)

    # ============================ 动作实现 ============================

    def _ai_poke_marker(self) -> str:
        return (self.config.get("ai_poke_marker") or "[戳一戳]").strip() or "[戳一戳]"

    @staticmethod
    def _clamp_float(value: Any) -> float:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return 0.0

    async def _poke_user(self, event: AstrMessageEvent, user_id: str,
                         group_id: str | None, times: int) -> bool:
        client = getattr(event, "bot", None)
        if client is None:
            logger.warning("[pokeplus] 当前平台不支持戳一戳(仅 OneBot v11)")
            return False
        return await self._poke_client(client, user_id, group_id, times)

    async def _poke_client(self, client, user_id: str, group_id: str | None, times: int) -> bool:
        try:
            uid = int(str(user_id).strip())
        except (TypeError, ValueError):
            return False
        if uid <= 0:
            return False
        interval = self._global_float("poke_interval", 0.3)
        for _ in range(max(1, times)):
            try:
                if group_id:
                    await client.call_action("group_poke", group_id=int(group_id), user_id=uid)
                else:
                    try:
                        await client.call_action("friend_poke", user_id=uid)
                    except Exception:
                        # 部分 OneBot 实现私聊戳用 send_poke
                        await client.call_action("send_poke", user_id=uid)
            except Exception as e:
                logger.warning(f"[pokeplus] 戳一戳发送失败 user_id={uid}: {e}")
                return False
            if interval > 0:
                await asyncio.sleep(interval)
        return True

    def _bundled_sticker_dir(self) -> str:
        """插件内置默认表情包目录（随插件分发）"""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "stickers_default")

    def _pick_sticker(self, section: dict) -> str | None:
        """从「内置默认表情包 + 用户配置图库」合并池中随机抽取一张"""
        paths: list[str] = []
        if section.get("use_default_stickers", True):
            paths.append(self._bundled_sticker_dir())
        paths.extend(str(p) for p in (section.get("sticker_paths") or []) if p)

        imgs: list[str] = []
        for p in paths:
            if os.path.isfile(p):
                imgs.append(p)
            elif os.path.isdir(p):
                for root, _dirs, files in os.walk(p):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                            imgs.append(os.path.join(root, f))
        return random.choice(imgs) if imgs else None

    async def _get_conversation(self, event: AstrMessageEvent):
        """获取或创建当前会话的 conversation, 使 LLM 回复使用人设并记入上下文"""
        try:
            conv_mgr = self.context.conversation_manager
            umo = event.unified_msg_origin
            cid = await conv_mgr.get_curr_conversation_id(umo)
            if not cid:
                cid = await conv_mgr.new_conversation(umo, event.get_platform_id())
            return await conv_mgr.get_conversation(umo, cid)
        except Exception as e:
            logger.warning(f"[pokeplus] 获取会话失败, 将以无上下文模式回复: {e}")
            return None

    def _cleanup_cooldowns(self, now: float):
        expire = max(
            600.0,
            self._global_float("user_cooldown", 5.0),
            self._global_float("session_cooldown", 5.0),
        )
        for m in (self._user_cd, self._session_cd):
            for k in [k for k, ts in m.items() if now - ts > expire]:
                del m[k]
