"""Telegram bot adapter — connects Telegram chats/topics to turnstone workstreams.

:class:`TurnstoneTelegramBot` uses python-telegram-bot with long polling so
no public URL or webhook endpoint is required.  Mirrors the Slack adapter
pattern.

Interaction model
-----------------
* **DMs**: every message is routed freely, no command needed.
* **Forum supergroups**: ``/turnstone`` creates a topic for the invoking
  user and waits for messages there.  Messages outside a bot-created
  topic are ignored.  Running ``/turnstone`` again in the same chat (by
  the same user) archives the old workstream and starts a fresh topic.
* **Regular groups**: ``/turnstone`` posts an opener message; messages
  replying anywhere in that session's reply chain (by the invoking user)
  are routed to it.

Note: Telegram's default bot privacy mode hides ordinary group messages
from bots.  Group sessions therefore require privacy mode to be disabled
(BotFather → /setprivacy → Disable); DMs are unaffected.

Events are consumed from the server's per-workstream SSE endpoint
(``GET /v1/api/workstreams/{ws_id}/events``) using httpx2's SSE support.
Inbound messages are sent directly to server nodes via HTTP
(``POST /v1/api/workstreams/{ws_id}/send``).

Install dependencies:
    pip install 'turnstone[telegram]'
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import httpx2
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from turnstone.channels._config import MAX_NOTIFY_TRACKING
from turnstone.channels._formatter import chunk_message
from turnstone.channels._routing import (
    MODEL_LOOKUP_FAILED_REPLY,
    ChannelRouter,
    get_cycle_entry,
    model_unavailable_reply,
    pop_cycle_entry,
    pop_ws_entries,
)
from turnstone.channels._sse import run_sse_stream
from turnstone.channels.telegram.routes import TelegramRoute
from turnstone.core.log import get_logger
from turnstone.sdk._types import TurnstoneAPIError
from turnstone.sdk.events import (
    ApprovalResolvedEvent,
    ApproveRequestEvent,
    ContentEvent,
    ErrorEvent,
    IntentVerdictEvent,
    ServerEvent,
    StreamEndEvent,
    ThinkingStartEvent,
    ThinkingStopEvent,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from telegram import InlineKeyboardMarkup, Message, Update
    from telegram.ext import ContextTypes, ExtBot

    from turnstone.channels.telegram.config import TelegramConfig
    from turnstone.core.storage._protocol import StorageBackend

log = get_logger(__name__)

_GREETING = "Hey! Let me know what I can help with."

# Telegram renders HTML; keep headroom below the 4096-character API limit
# for the <b>/<i>/<code>/<pre> tags the renderer adds on top of the
# already-escaped text.
_SEND_LIMIT_SLACK: int = 96
# Inbound-message limits — one user can't exhaust the LLM budget on
# their own.
_PER_USER_RATE_WINDOW_S: float = 60.0
_PER_USER_RATE_LIMIT: int = 10
_PER_USER_RATE_CAP: int = 4096  # LRU bound on the per-user deque map
_MAX_INBOUND_MESSAGE_LEN: int = 8192  # chars; roughly 8 KiB

# /link is the only flow that reads Turnstone API tokens out of user
# input; throttle aggressively so throw-away Telegram accounts can't
# online-enumerate Turnstone API tokens (mirrors Slack / Discord).
_LINK_RATE_WINDOW_S: float = 3600.0
_LINK_RATE_LIMIT: int = 5
_LINK_RATE_CAP: int = 2048

# Reply-chain sessions (regular groups) route by membership in the
# session's message set: the opener, the bot's responses, and accepted
# inbound messages.  Bounded so a long session can't grow it unboundedly.
_SESSION_MSG_CAP: int = 512

# Approval text budget — the marker line and judge verdicts ride on the
# same message, so the tool list alone can't crowd them past the limit.
_APPROVAL_TEXT_BUDGET: int = 3400
_APPROVAL_PER_ITEM_PREVIEW: int = 600

_TOPIC_NAME_MAX: int = 100

# Approval prompts embed their correlation data in the message text
# (Discord's embed-footer pattern) because Telegram caps callback_data
# at 64 bytes — too small for ws_id + cycle_id + owner.  The line is
# stripped from nothing: parse it back out of the plain text Telegram
# returns on callback queries.
_APPROVAL_MARKER_RE = re.compile(r"^ts:([^|\n]+)\|([^|\n]*)\|([^|\n]*)$", re.MULTILINE)


def _escape_html(text: str) -> str:
    """Escape ``&<>`` in user- or model-authored text bound for Telegram HTML.

    Telegram's parse mode is HTML, so ``&``, ``<`` and ``>`` become
    entities.  That one move neutralizes tag injection and keeps a
    literal ``@username`` — which Telegram resolves anywhere — but the
    adapter only ever interpolates escaped text into framing it authors
    itself.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _render_escaped_html(text: str) -> str:
    """Convert a *pre-escaped* model message into display HTML.

    Input must already be HTML-escaped (:func:`_escape_html`).  Fenced
    code blocks become ``<pre>`` and inline code spans ``<code>``;
    ``**bold**`` and ``*italic*`` markers outside code become ``<b>`` /
    ``<i>``.  Unpaired markers are left literal so malformed model
    output can never unbalance the markup.  Bot-authored framing is
    never passed through this — it is constructed directly.
    """
    parts = text.split("```")
    out: list[str] = []
    for i, seg in enumerate(parts):
        if i % 2 == 1:
            # Code fence: no inline markers inside.
            out.append(f"<pre>{seg}</pre>")
            continue
        # Inline code spans first so bold/italic never rewrite inside them.
        chunks = seg.split("`")
        for j, span in enumerate(chunks):
            if j % 2 == 1:
                out.append(f"<code>{span}</code>")
                continue
            span = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", span, flags=re.DOTALL)
            span = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", span)
            out.append(span)
    return "".join(out)


def _render_html(text: str) -> str:
    """Escape *text* for Telegram and convert its light markdown to HTML."""
    return _render_escaped_html(_escape_html(text))


def _safe_html_cut(text: str, limit: int) -> str:
    """Cut escaped HTML *text* to *limit* characters without splitting an entity.

    The display window is truncated, so a cut can land inside ``&amp;``.
    A partial entity at the tail (``&`` followed by fragment characters
    with no closing ``;``) would fail Telegram's HTML parser, so strip
    that trailing fragment.  A complete entity like ``&amp;`` keeps its
    ``;`` and is left alone.
    """
    cut = text[:limit]
    return re.sub(r"&[a-zA-Z0-9#]*$", "", cut)


def _code_name(raw: object) -> str:
    """A tool's name for a Telegram code span: one line, at most 100 characters.

    An unknown tool's name is whatever the model sent, so the span carries
    it flattened, with no backtick of its own to end the span early.
    """
    from turnstone.channels._formatter import truncate

    flat = " ".join(str(raw).splitlines()).replace("`", "'").strip()
    return truncate(flat or "tool", 100)


# Workstream names rendered in the console's workstream browser.
# Readable labels (where + who), not raw channel IDs — the ws_id
# already carries that.
_NAME_PART_MAX = 40


def _session_name(chat_title: str, full_name: str) -> str:
    """Workstream name for a group session: ``Telegram: <group> — <user>``."""
    title = (chat_title or "").strip()[:_NAME_PART_MAX] or "group"
    who = (full_name or "").strip()[:_NAME_PART_MAX] or "user"
    return f"Telegram: {title} — {who}"


def _dm_name(full_name: str) -> str:
    """Workstream name for a DM session: ``Telegram DM: <user>``."""
    who = (full_name or "").strip()[:_NAME_PART_MAX] or "user"
    return f"Telegram DM: {who}"


def _parse_approval_marker(text: str | None) -> tuple[str, str, str] | None:
    """Extract ``(ws_id, cycle_id, owner_id)`` from an approval message.

    The marker is the last ``ts:...`` line the bot appended at post
    time.  Pre-multi-cycle prompts carry an empty cycle_id; the shared
    approval bookkeeping falls back to the workstream's single tracked
    entry in that case.
    """
    if not text:
        return None
    match = _APPROVAL_MARKER_RE.search(text)
    if match is None:
        return None
    return match.group(1), match.group(2), match.group(3)


@dataclass(frozen=True)
class _PendingApproval:
    """One posted approval prompt awaiting a decision."""

    chat_id: int
    message_id: int
    owner_user_id: str | None = None
    # Bot-authored HTML body as posted, including the correlation
    # marker line, so IntentVerdictEvent can re-edit the message
    # without reconstructing it.
    body_html: str = ""
    # Approval-cycle identity + member call_ids.  A workstream can hold
    # several concurrent cycles (parallel task agents), each its own
    # Telegram message; the cycle_id routes the button click to exactly
    # this round and call_ids route IntentVerdictEvents (keyed by
    # call_id) onto the right message.  Verdict sections accumulate so
    # repeat events stack instead of overwriting each other.
    cycle_id: str = ""
    call_ids: frozenset[str] = frozenset()
    verdicts: tuple[str, ...] = ()


@dataclass
class StreamingMessage:
    """Accumulates streamed content and periodically edits a Telegram message."""

    bot: Any
    chat_id: int
    thread_id: str = ""
    # Pre-resolved send placement ``(message_thread_id,
    # reply_to_message_id)`` for new messages (forum topic, reply-chain
    # anchor, or ``(0, 0)`` for DMs).  Resolved by the bot at
    # construction time from its forum cache; edits never need it.
    thread: tuple[int, int] = (0, 0)
    max_length: int = 4096
    edit_interval: float = 1.5

    _message_id: int | None = field(default=None, init=False, repr=False)
    _buffer: list[str] = field(default_factory=list, init=False, repr=False)
    # Rolling in-progress display string; stops growing at max_length so
    # per-flush cost is O(max_length) instead of O(total_streamed_chars).
    _display: str = field(default="", init=False, repr=False)
    _last_edit: float = field(default=0.0, init=False, repr=False)
    _last_text: str = field(default="", init=False, repr=False)
    _finalized_text: str | None = field(default=None, init=False, repr=False)

    @property
    def message_id(self) -> int | None:
        """Telegram message ID of the initial posted message, if sent."""
        return self._message_id

    @property
    def accumulated_text(self) -> str:
        """The joined text of everything appended so far.

        Cached after ``finalize()`` so downstream callers that read the
        full message content don't re-join the buffer twice.
        """
        if self._finalized_text is not None:
            return self._finalized_text
        return "".join(self._buffer)

    async def append(self, text: str) -> None:
        self._buffer.append(text)
        if len(self._display) < self.max_length:
            display = _escape_html("".join(self._buffer))
            self._display = _safe_html_cut(display, self.max_length)
        if time.monotonic() - self._last_edit >= self.edit_interval:
            await self._flush()

    async def finalize(self) -> None:
        content = "".join(self._buffer)
        self._finalized_text = content
        if not content:
            return

        # Chunk the escaped text so HTML-entity expansion is already
        # counted against the limit; rendering per chunk only adds the
        # (bounded) tag overhead the headroom covers.
        chunks = chunk_message(_escape_html(content), self.max_length - _SEND_LIMIT_SLACK)
        rendered = [_render_escaped_html(chunk) for chunk in chunks]
        if self._message_id is not None:
            try:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self._message_id,
                    text=rendered[0],
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                log.debug("telegram.streaming_message.finalize_edit_failed", exc_info=True)
            for text in rendered[1:]:
                await _send_html(
                    self.bot, self.chat_id, text, self.thread, max_length=self.max_length
                )
        else:
            for text in rendered:
                message = await _send_html(
                    self.bot, self.chat_id, text, self.thread, max_length=self.max_length
                )
                if self._message_id is None:
                    self._message_id = message.message_id

    async def _flush(self) -> None:
        display = self._display
        if not display or display == self._last_text:
            return

        try:
            if self._message_id is None:
                message = await _send_html(self.bot, self.chat_id, display, self.thread)
                self._message_id = message.message_id
            else:
                await self.bot.edit_message_text(
                    chat_id=self.chat_id,
                    message_id=self._message_id,
                    text=display,
                    parse_mode=ParseMode.HTML,
                )
        except BadRequest as exc:
            # Telegram rejects no-op edits ("message is not modified")
            # and messages whose text it deems malformed; the next
            # flush retries with new content either way.
            log.debug("telegram.streaming_message.flush_bad_request", error=str(exc))
        except Exception:
            log.debug("telegram.streaming_message.flush_failed", exc_info=True)

        self._last_text = display
        self._last_edit = time.monotonic()

    def _thread_kwargs(self) -> tuple[int, int]:
        return self.thread


async def _send_html(
    bot: ExtBot[Any],
    chat_id: int,
    text: str,
    thread: tuple[int, int] = (0, 0),
    *,
    reply_markup: InlineKeyboardMarkup | None = None,
    max_length: int = 4096,
) -> Message:
    """Send one HTML chunk to *chat_id*, returning the new message.

    ``thread`` is ``(message_thread_id, reply_to_message_id)`` with ``0``
    meaning absent — Telegram rejects the ``None`` keyword values the
    typed signature suggests, so the tuple keeps call sites simple.

    If Telegram's HTML parser rejects the chunk (a pathological entity
    split is the only realistic trigger), retry it unformatted rather
    than dropping the content.
    """
    message_thread_id = thread[0] or None
    reply_to_message_id = thread[1] or None
    try:
        return await bot.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode=ParseMode.HTML,
            reply_markup=reply_markup,
            message_thread_id=message_thread_id,
            reply_to_message_id=reply_to_message_id,
        )
    except BadRequest as exc:
        log.warning("telegram.send_html_fallback_plain", chat_id=chat_id, error=str(exc))
        return await bot.send_message(
            chat_id=chat_id,
            text=text[:max_length],
            message_thread_id=message_thread_id,
            reply_to_message_id=reply_to_message_id,
        )


class TurnstoneTelegramBot:
    """Telegram bot bridging Telegram chats/topics to turnstone workstreams."""

    channel_type: str = "telegram"
    _MAX_NOTIFY_TRACKING: int = MAX_NOTIFY_TRACKING

    def __init__(
        self,
        config: TelegramConfig,
        server_url: str,
        storage: StorageBackend,
        *,
        api_token: str = "",
        console_url: str = "",
        console_token_factory: Callable[[], str] | None = None,
        server_token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self._server_url = server_url.rstrip("/")
        self._console_url = console_url.rstrip("/") if console_url else ""
        # Each SSE connection to a server node authenticates through this. A static token goes
        # the same way, so the shared client holds no credential.
        self._token_factory = server_token_factory or ((lambda: api_token) if api_token else None)
        self.storage = storage

        self.router = ChannelRouter(
            server_url,
            storage,
            auto_approve=config.auto_approve,
            auto_approve_tools=list(config.auto_approve_tools),
            skill=config.skill,
            api_token=api_token,
            console_url=console_url,
            console_token_factory=console_token_factory,
            server_token_factory=server_token_factory,
        )

        self._subscribed_ws: set[str] = set()
        self._sse_tasks: dict[str, asyncio.Task[None]] = {}
        self._streaming: dict[str, StreamingMessage] = {}

        # Keyed by (ws_id, cycle_id) — one entry per approval MESSAGE.
        self._pending_approval: dict[tuple[str, str], _PendingApproval] = {}
        self._notify_ws_map: dict[tuple[int, int], tuple[str, TelegramRoute]] = {}
        # Per-workstream override used to route the next streamed assistant
        # response back into a notification reply chain/topic instead of the
        # default session thread.
        self._notify_reply_routes: dict[str, TelegramRoute] = {}
        self._channel_sessions: dict[tuple[str, str], tuple[str, str]] = {}
        # Reply-chain session membership (regular groups): session key ->
        # message IDs that belong to the chain, insertion-ordered.
        self._session_msgs: dict[tuple[str, str], OrderedDict[int, None]] = {}
        # chat_id -> is_forum, learned from inbound updates and get_chat.
        self._forum_cache: dict[int, bool] = {}
        # Per-user sliding-window rate limit; maps telegram_user_id -> deque
        # of recent allow timestamps within _PER_USER_RATE_WINDOW_S.
        self._rate_buckets: OrderedDict[str, deque[float]] = OrderedDict()
        # Separate sliding-window throttle for /link so an abusive user
        # can't enumerate tokens while still being allowed to chat at the
        # baseline message rate.
        self._link_buckets: OrderedDict[str, deque[float]] = OrderedDict()

        self._http_client = httpx2.AsyncClient(
            timeout=httpx2.Timeout(connect=10.0, read=90.0, write=10.0, pool=10.0),
        )

        self._app = Application.builder().token(config.bot_token).build()
        self._app.add_handler(CommandHandler("turnstone", self._on_turnstone_command))
        self._app.add_handler(CommandHandler("link", self._on_link_command))
        self._app.add_handler(CommandHandler("unlink", self._on_unlink_command))
        self._app.add_handler(CallbackQueryHandler(self._on_callback))
        self._app.add_handler(
            MessageHandler(
                filters.UpdateType.MESSAGE & filters.TEXT & ~filters.COMMAND, self._on_message
            )
        )
        self._app.add_error_handler(self._on_error)

        self._started = False

    # -- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        await self._app.initialize()
        await self._app.bot.set_my_commands(
            [
                ("turnstone", "Start or restart a turnstone session"),
                ("link", "Link your Telegram account to Turnstone"),
                ("unlink", "Unlink your Telegram account"),
            ]
        )
        await self._recover_routes()
        await self._app.start()
        assert self._app.updater is not None  # built without updater=None
        await self._app.updater.start_polling(
            allowed_updates=["message", "callback_query"],
            drop_pending_updates=True,
        )
        self._started = True
        log.info("telegram.starting_long_polling")

    async def stop(self) -> None:
        for ws_id in list(self._subscribed_ws):
            await self.unsubscribe_ws(ws_id)
        await self.router.aclose()
        await self._http_client.aclose()
        if self._started:
            assert self._app.updater is not None
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._started = False
        log.info("telegram.stopped")

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.warning("telegram.handler_error", error=str(context.error))

    async def _recover_routes(self) -> None:
        """Restore all associations, then subscribe only to already-live sessions."""
        routes = await asyncio.to_thread(
            self.storage.list_channel_routes_by_type,
            "telegram",
        )

        latest_sessions: dict[tuple[str, str], tuple[str, str]] = {}

        for route in routes:
            ws_id = route["ws_id"]
            channel_id = route["channel_id"]

            telegram_route = TelegramRoute.parse(channel_id)
            if telegram_route.chat and telegram_route.user_id and telegram_route.thread_id:
                key = (telegram_route.chat, telegram_route.user_id)
                existing = latest_sessions.get(key)
                # Telegram message IDs increase within a chat, so the
                # larger thread anchor is the newest session.
                try:
                    if existing is None or int(telegram_route.thread_id) > int(existing[1]):
                        latest_sessions[key] = (ws_id, telegram_route.thread_id)
                except ValueError:
                    log.warning(
                        "telegram.route_recover_bad_thread",
                        ws_id=ws_id,
                        channel_id=channel_id,
                    )

        self._channel_sessions = latest_sessions

        live_ids = await self.router.get_live_workstream_ids(route["ws_id"] for route in routes)
        for route in routes:
            if route["ws_id"] in live_ids:
                await self.subscribe_ws(route["ws_id"], route["channel_id"])
                log.info(
                    "telegram.route_recovered", ws_id=route["ws_id"], channel_id=route["channel_id"]
                )

        for (chat_id, user_id), (ws_id, thread_id) in self._channel_sessions.items():
            log.info(
                "telegram.session_recovered",
                chat_id=chat_id,
                user_id=user_id,
                ws_id=ws_id,
                thread_id=thread_id,
            )

    # -- identity ------------------------------------------------------------

    async def _require_linked(self, chat_id: int, telegram_user_id: str, *, private: bool) -> bool:
        """Return True if *telegram_user_id* is mapped to a Turnstone account.

        Posts a hint on False so the caller can early-return.  /link is
        DM-only (a token typed in a group command is visible to every
        member), so the hint points there from groups.
        """
        mapped = await self.router.resolve_user("telegram", telegram_user_id)
        if mapped:
            return True
        if private:
            hint = (
                "You need to link your Telegram account to a Turnstone user before "
                "using this bot.  Run /link here — tokens are managed in the "
                "Turnstone console."
            )
        else:
            hint = (
                "You need to link your Telegram account to a Turnstone user before "
                "using this bot.  Open this bot's profile and send /link in our "
                "private chat."
            )
        await self._reply(chat_id, hint)
        return False

    def _allow_user_send(self, telegram_user_id: str) -> bool:
        """Return True when *telegram_user_id* is within the send rate limit.

        Sliding-window counter: up to ``_PER_USER_RATE_LIMIT`` allowed
        sends per ``_PER_USER_RATE_WINDOW_S``.  Stale entries are evicted
        lazily, and the bucket map is LRU-bounded so a chat with churning
        user IDs can't grow the map unboundedly.
        """
        now = time.monotonic()
        window_start = now - _PER_USER_RATE_WINDOW_S
        bucket = self._rate_buckets.get(telegram_user_id)
        if bucket is None:
            bucket = deque()
            self._rate_buckets[telegram_user_id] = bucket
            while len(self._rate_buckets) > _PER_USER_RATE_CAP:
                self._rate_buckets.popitem(last=False)
        else:
            self._rate_buckets.move_to_end(telegram_user_id)
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= _PER_USER_RATE_LIMIT:
            return False
        bucket.append(now)
        return True

    async def _check_inbound(
        self,
        chat_id: int,
        telegram_user_id: str,
        text: str,
    ) -> bool:
        """Validate an inbound message against size + rate limits.

        Returns True when the caller should proceed.  Replies in-chat on
        rejection so the user understands why their message was dropped.
        """
        if len(text) > _MAX_INBOUND_MESSAGE_LEN:
            await self._reply(
                chat_id,
                f"Message too long ({len(text)} chars).  Please keep "
                f"messages under {_MAX_INBOUND_MESSAGE_LEN} characters.",
            )
            log.info(
                "telegram.inbound_oversize",
                user_id=telegram_user_id,
                length=len(text),
            )
            return False
        if not self._allow_user_send(telegram_user_id):
            await self._reply(
                chat_id,
                f"You're sending messages too fast.  Limit is "
                f"{_PER_USER_RATE_LIMIT} per "
                f"{int(_PER_USER_RATE_WINDOW_S)} seconds.",
            )
            log.warning("telegram.inbound_rate_limited", user_id=telegram_user_id)
            return False
        return True

    def _allow_link_attempt(self, telegram_user_id: str) -> bool:
        """Return True when *telegram_user_id* is under the /link rate limit.

        Mirrors the Slack helper at ``slack/bot.py::_allow_link_attempt``
        — 5 attempts per hour per Telegram user, LRU-bounded.
        """
        now = time.monotonic()
        window_start = now - _LINK_RATE_WINDOW_S
        bucket = self._link_buckets.get(telegram_user_id)
        if bucket is None:
            bucket = deque()
            self._link_buckets[telegram_user_id] = bucket
            while len(self._link_buckets) > _LINK_RATE_CAP:
                self._link_buckets.popitem(last=False)
        else:
            self._link_buckets.move_to_end(telegram_user_id)
        while bucket and bucket[0] < window_start:
            bucket.popleft()
        if len(bucket) >= _LINK_RATE_LIMIT:
            return False
        bucket.append(now)
        return True

    async def _on_link_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Link the caller's Telegram ID to the Turnstone user that owns *token*.

        DM-only: a token typed into a group command is visible to every
        member of the group.
        """
        from turnstone.core.auth import hash_token

        message = update.effective_message
        user = update.effective_user
        chat = update.effective_chat
        if message is None or user is None or chat is None:
            return
        telegram_user_id = str(user.id)

        if chat.type != "private":
            await message.reply_text("Please run /link in a private chat with me.")
            return

        token = " ".join(context.args).strip() if context.args else ""

        if not self._allow_link_attempt(telegram_user_id):
            log.warning("telegram.link_rate_limited", telegram_user_id=telegram_user_id)
            await message.reply_text(
                f"Too many /link attempts.  Try again later — limit is {_LINK_RATE_LIMIT} per hour."
            )
            return

        if not token:
            await message.reply_text("Usage: /link <your_api_token>")
            return

        existing = await asyncio.to_thread(
            self.storage.get_channel_user, "telegram", telegram_user_id
        )
        if existing:
            await message.reply_text("Your Telegram account is already linked.  Use /unlink first.")
            return

        record = await asyncio.to_thread(self.storage.get_api_token_by_hash, hash_token(token))
        if record is None or not record.get("user_id"):
            # Generic failure message — don't leak whether the token
            # existed but was e.g. revoked.
            await message.reply_text("Invalid token.  Please provide a valid Turnstone API token.")
            log.info("telegram.link_invalid_token", telegram_user_id=telegram_user_id)
            return

        await asyncio.to_thread(
            self.storage.create_channel_user,
            "telegram",
            telegram_user_id,
            record["user_id"],
        )
        await message.reply_text("Linked.  You can now use Turnstone from Telegram.")
        log.info(
            "telegram.link_succeeded",
            telegram_user_id=telegram_user_id,
            turnstone_user_id=record["user_id"],
        )

    async def _on_unlink_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Drop the Telegram → Turnstone mapping for the caller."""
        user = update.effective_user
        message = update.effective_message
        if user is None or message is None:
            return
        telegram_user_id = str(user.id)
        existing = await asyncio.to_thread(
            self.storage.get_channel_user, "telegram", telegram_user_id
        )
        if not existing:
            await message.reply_text("Your Telegram account was not linked.")
            return
        await asyncio.to_thread(self.storage.delete_channel_user, "telegram", telegram_user_id)
        await message.reply_text("Unlinked.")
        log.info("telegram.unlink_succeeded", telegram_user_id=telegram_user_id)

    # -- commands / session lifecycle ----------------------------------------

    async def _on_turnstone_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Start or restart a per-user session in this chat.

        - Forum supergroups: creates a topic for the session.
        - Regular groups: posts an opener message that anchors the
          session's reply chain.
        - Private chats: no anchor is needed; messages route freely.
        """
        message = update.effective_message
        user = update.effective_user
        chat = update.effective_chat
        if message is None or user is None or chat is None:
            return

        chat_id = str(chat.id)
        user_id = str(user.id)
        private = chat.type == "private"
        self._forum_cache[chat.id] = bool(getattr(chat, "is_forum", False))

        text = " ".join(context.args).strip() if context.args else ""

        if not private and self.config.allowed_chats and chat.id not in self.config.allowed_chats:
            await message.reply_text("Sorry, turnstone isn't enabled in this chat.")
            return

        # Only linked Telegram users can start sessions — otherwise every
        # group member would be creating workstreams under the shared
        # gateway identity with no attribution.
        if not await self._require_linked(chat.id, user_id, private=private):
            return

        # The channel adapter model (the Roles tab's), else the server default.
        # Read it before touching the current session, so a failed read leaves
        # that session in place instead of archiving it for one that can't start.
        try:
            channel_model = await self.router.get_channel_default_alias()
        except Exception:
            log.warning("telegram.channel_model_lookup_failed", exc_info=True)
            await message.reply_text(MODEL_LOOKUP_FAILED_REPLY)
            return

        # Create the anchor first so a failure (missing Manage Topics
        # permission) leaves the previous session intact.
        thread_id = ""
        if not private:
            try:
                if self._forum_cache.get(chat.id, False):
                    topic_name = text[:_TOPIC_NAME_MAX] or f"turnstone — {user.full_name}"
                    topic = await self._app.bot.create_forum_topic(chat.id, name=topic_name)
                    thread_id = str(topic.message_thread_id)
                else:
                    opener = await self._app.bot.send_message(
                        chat.id,
                        f"{_escape_html(user.full_name)} started a turnstone session.",
                    )
                    thread_id = str(opener.message_id)
                    self._track_session_msg((chat_id, user_id), opener.message_id)
            except Exception:
                log.warning("telegram.session_anchor_failed", chat_id=chat.id, exc_info=True)
                await message.reply_text(
                    "I couldn't open a conversation here. For forum groups, check that "
                    "I have the Manage Topics permission."
                )
                return

            existing = self._channel_sessions.get((chat_id, user_id))
            if existing is not None:
                old_ws_id, old_thread_id = existing
                try:
                    await self._archive_session(chat.id, user_id, old_ws_id, old_thread_id)
                except Exception:
                    log.warning("telegram.session_archive_failed", exc_info=True)
                    await message.reply_text(
                        "I couldn't close the previous session. Please try again."
                    )
                    return

            greeting_kwargs = await self._thread_target(chat.id, thread_id)
            try:
                greeting = await _send_html(self._app.bot, chat.id, _GREETING, greeting_kwargs)
                if not self._forum_cache.get(chat.id, False):
                    self._track_session_msg((chat_id, user_id), greeting.message_id)
            except Exception:
                log.debug("telegram.greeting_failed", chat_id=chat.id, exc_info=True)

        route = TelegramRoute(chat=chat_id, user_id=user_id, thread_id=thread_id or None)

        try:
            ws_id, _ = await self.router.get_or_create_workstream(
                channel_type="telegram",
                channel_id=route.to_channel_id(),
                name=_session_name(getattr(chat, "title", ""), user.full_name),
                # A model that isn't loaded fails the create.
                model=channel_model,
                client_type="chat",
            )
        except Exception as exc:
            reply = model_unavailable_reply(exc)
            if reply is None:
                raise
            await self._reply(chat.id, reply, thread_id=thread_id)
            return
        await self.subscribe_ws(ws_id, route.to_channel_id())
        self._channel_sessions[(chat_id, user_id)] = (ws_id, thread_id)

        if text:
            await self.router.send_message(ws_id, text)

        log.info(
            "telegram.session_started",
            ws_id=ws_id,
            chat=chat_id,
            user=user_id,
            thread_id=thread_id,
        )

    async def _archive_session(
        self,
        chat_id: int,
        user_id: str,
        ws_id: str,
        thread_id: str,
    ) -> None:
        """Close the selected session without removing a concurrent replacement."""
        route = TelegramRoute(chat=str(chat_id), user_id=user_id, thread_id=thread_id or None)
        await self.router.close_workstream(ws_id)
        if not await self.router.delete_route(
            "telegram", route.to_channel_id(), expected_ws_id=ws_id
        ):
            current = await asyncio.to_thread(
                self.storage.get_channel_route, "telegram", route.to_channel_id()
            )
            key = (str(chat_id), user_id)
            if self._channel_sessions.get(key) == (ws_id, thread_id):
                if current is None:
                    self._channel_sessions.pop(key, None)
                else:
                    self._channel_sessions[key] = (current["ws_id"], thread_id)
            raise RuntimeError("Telegram session changed; retry closing it")
        if self._channel_sessions.get((str(chat_id), user_id)) == (ws_id, thread_id):
            self._channel_sessions.pop((str(chat_id), user_id), None)
        self._session_msgs.pop((str(chat_id), user_id), None)
        await self.unsubscribe_ws(ws_id)
        try:
            await self._reply(
                chat_id, "<i>This session has been archived.</i>", thread_id=thread_id
            )
        except Exception:
            log.debug("telegram.archive_session.notify_failed", ws_id=ws_id, exc_info=True)
        log.info("telegram.session_archived", ws_id=ws_id, chat=chat_id, user=user_id)

    # -- inbound messages ----------------------------------------------------

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Route messages — free in DMs, session-scoped in groups."""
        message = update.effective_message
        user = update.effective_user
        chat = update.effective_chat
        if message is None or user is None or chat is None or user.is_bot:
            return

        chat_id = str(chat.id)
        user_id = str(user.id)
        text = message.text.strip() if message.text else ""
        if not text:
            return

        self._forum_cache[chat.id] = bool(getattr(chat, "is_forum", False))
        reply = message.reply_to_message
        reply_id = reply.message_id if reply is not None else None

        # Replies to tracked notifications route back to their origin
        # workstream, in DMs and groups alike.
        if reply_id is not None:
            entry = self._notify_ws_map.get((chat.id, reply_id))
            if entry is not None:
                await self._handle_notification_reply(
                    chat.id, user_id, text, reply_id, message.message_thread_id, entry
                )
                return

        if chat.type == "private":
            await self._handle_dm(chat.id, user_id, text, user.full_name)
            return

        if self.config.allowed_chats and chat.id not in self.config.allowed_chats:
            return

        session = self._channel_sessions.get((chat_id, user_id))
        if session is None:
            return

        ws_id, thread_id = session
        if not await self._session_message_matches(
            chat.id, thread_id, message.message_thread_id, reply_id, (chat_id, user_id)
        ):
            return

        # Gate message routing on a linked Telegram user so all LLM usage
        # can be attributed to a Turnstone account.  Unlinked users get a
        # pointer to /link in the bot's private chat.
        if not await self._require_linked(chat.id, user_id, private=False):
            return

        if not await self._check_inbound(chat.id, user_id, text):
            return

        try:
            route = TelegramRoute(chat=chat_id, user_id=user_id, thread_id=thread_id or None)
            recovered_ws_id, _is_new = await self.router.get_or_create_workstream(
                channel_type="telegram",
                channel_id=route.to_channel_id(),
                name=_session_name(getattr(chat, "title", ""), user.full_name),
                client_type="chat",
                require_existing=True,
            )
            if recovered_ws_id != ws_id:
                await self.unsubscribe_ws(ws_id)
            current = self._channel_sessions.get((chat_id, user_id))
            if current is None or current[1] != thread_id:
                raise RuntimeError("Telegram session changed; retry your message")
            ws_id = recovered_ws_id
            self._channel_sessions[(chat_id, user_id)] = (ws_id, thread_id)
            await self.subscribe_ws(ws_id, route.to_channel_id())
            self._track_session_msg((chat_id, user_id), message.message_id)
            await self.router.send_message(ws_id, text)
            log.info("telegram.message_dispatched", ws_id=ws_id, chat=chat_id)
        except Exception as exc:
            log.exception("telegram.message_dispatch_failed", chat=chat_id)
            error_reply = (
                model_unavailable_reply(exc) or "Sorry, something went wrong routing your message."
            )
            await self._reply(chat.id, _escape_html(error_reply), thread_id=thread_id)

    async def _session_message_matches(
        self,
        chat_id: int,
        thread_id: str,
        message_thread_id: int | None,
        reply_id: int | None,
        session_key: tuple[str, str],
    ) -> bool:
        """Whether an inbound group message belongs to the caller's session.

        Forum topics match by ``message_thread_id``; regular groups match
        by membership in the session's reply chain.
        """
        if self._forum_cache.get(chat_id, False):
            return message_thread_id is not None and str(message_thread_id) == thread_id
        if reply_id is None:
            return False
        tracked = self._session_msgs.get(session_key)
        return tracked is not None and reply_id in tracked

    async def _handle_dm(self, chat_id: int, user_id: str, text: str, full_name: str = "") -> None:
        """Handle a direct message — no command required."""
        # Same identity gate as /turnstone and group messages: a
        # Telegram user must be linked to a Turnstone account before
        # they can drive LLM calls via DM.
        if not await self._require_linked(chat_id, user_id, private=True):
            return

        if not await self._check_inbound(chat_id, user_id, text):
            return

        # DMs have no threading, so the route is keyed by (chat, user)
        # alone.  Replying to an old notification still lands here when
        # its tracking entry has been evicted, continuing the session.
        route = TelegramRoute(chat=str(chat_id), user_id=user_id, thread_id=None)

        try:
            ws_id, _is_new = await self.router.get_or_create_workstream(
                channel_type="telegram",
                channel_id=route.to_channel_id(),
                name=_dm_name(full_name),
                # Looked up only when this starts or forks a conversation: a DM
                # to a live one doesn't need it.
                model=self.router.get_channel_default_alias,
                client_type="chat",
            )
            await self.subscribe_ws(ws_id, route.to_channel_id())

            await self.router.send_message(ws_id, text)
            log.info("telegram.dm_dispatched", ws_id=ws_id, user=user_id)
        except Exception as exc:
            log.exception("telegram.dm_dispatch_failed", user=user_id)
            error_reply = (
                model_unavailable_reply(exc) or "Sorry, something went wrong routing your message."
            )
            await self._reply(chat_id, _escape_html(error_reply))

    async def _handle_notification_reply(
        self,
        chat_id: int,
        user_id: str,
        text: str,
        reply_id: int,
        message_thread_id: int | None,
        entry: tuple[str, TelegramRoute],
    ) -> None:
        """Route a reply to a tracked notification back to its workstream."""
        origin_ws_id, notify_route = entry
        if str(chat_id) != notify_route.chat or user_id != (notify_route.user_id or ""):
            return
        # Same identity + rate/size gates the other two routing paths
        # use — the author-match above narrows the surface to the
        # originally notified user, but that user may have unlinked or
        # their token may have been revoked since the notification was
        # recorded.
        if not await self._require_linked(chat_id, user_id, private=True):
            return
        if not await self._check_inbound(chat_id, user_id, text):
            return
        # The next streamed response follows the notification: into the
        # same topic in forum groups, or as a reply into the same chain
        # anchored on the notification message in regular groups.
        forum = self._forum_cache.get(chat_id, False)
        reply_route = TelegramRoute(
            chat=str(chat_id),
            user_id=user_id,
            thread_id=str(message_thread_id) if forum and message_thread_id else str(reply_id),
        )
        self._notify_reply_routes[origin_ws_id] = reply_route
        log.info(
            "telegram.notification_reply_attempt",
            reply_id=reply_id,
            ws_id=origin_ws_id,
            chat=chat_id,
            user=user_id,
        )
        try:
            await self.router.send_message(origin_ws_id, text)
            log.info("telegram.notification_reply_routed", ws_id=origin_ws_id)
        except TurnstoneAPIError as exc:
            self._notify_reply_routes.pop(origin_ws_id, None)
            if exc.status_code == 404:
                self._clear_notification_tracking_for_ws(origin_ws_id)
                log.warning(
                    "telegram.notification_reply_dead_ws",
                    ws_id=origin_ws_id,
                    reply_id=reply_id,
                )
                try:
                    await self._reply(
                        chat_id,
                        "This notification is no longer linked to an active session. "
                        "Please start a new one with /turnstone.",
                    )
                except Exception:
                    log.debug(
                        "telegram.notification_reply_dead_ws_notice_failed",
                        exc_info=True,
                    )
            else:
                log.exception("telegram.notification_reply_failed")
        except Exception:
            self._notify_reply_routes.pop(origin_ws_id, None)
            log.exception("telegram.notification_reply_failed")

    # -- subscription management ---------------------------------------------

    async def subscribe_ws(self, ws_id: str, channel_id: str) -> None:
        # Recover from a dead SSE task before the early-return check.  If the
        # prior listener died with an unhandled exception, its ws_id is still
        # in _subscribed_ws — without this purge subscribe_ws would silently
        # no-op forever.
        prior = self._sse_tasks.get(ws_id)
        if prior is not None and prior.done():
            log.info("telegram.sse_task_recovered", ws_id=ws_id)
            self._sse_tasks.pop(ws_id, None)
            self._subscribed_ws.discard(ws_id)

        if ws_id in self._subscribed_ws:
            return

        task = asyncio.create_task(
            self._sse_listener(ws_id, channel_id),
            name=f"sse:{ws_id}",
        )
        self._sse_tasks[ws_id] = task
        self._subscribed_ws.add(ws_id)
        log.info("telegram.subscribed", ws_id=ws_id, channel_id=channel_id)

    def _pop_ws_approvals(self, ws_id: str) -> None:
        """Drop every tracked approval message for *ws_id* (all cycles)."""
        pop_ws_entries(self._pending_approval, ws_id)

    def _clear_ws_state(self, ws_id: str) -> None:
        """Drop all in-memory state keyed by *ws_id*.

        Does not cancel or await the SSE task — callers handle task
        lifecycle differently (``unsubscribe_ws`` cancels and awaits;
        ``_stop_unavailable_stream`` is itself invoked from inside the task).
        """
        self._subscribed_ws.discard(ws_id)
        self._streaming.pop(ws_id, None)
        self._pop_ws_approvals(ws_id)
        self._clear_notification_tracking_for_ws(ws_id)

    async def unsubscribe_ws(self, ws_id: str) -> None:
        task = self._sse_tasks.pop(ws_id, None)
        if task is not None:
            task.cancel()
            # Await the cancelled task so CancelledError propagates out of
            # the SSE loop before we clear the per-ws state below.
            # CancelledError is expected; other exceptions from the SSE
            # loop are already logged there and must not block shutdown.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        self._clear_ws_state(ws_id)

        log.info("telegram.unsubscribed", ws_id=ws_id)

    async def _sse_listener(self, ws_id: str, channel_id: str) -> None:
        """Connect to the server SSE endpoint and dispatch events.

        Delegates the reconnect/backoff loop to :func:`run_sse_stream`.
        """

        async def _on_event(event: ServerEvent) -> None:
            effective_route = self._notify_reply_routes.get(ws_id) or TelegramRoute.parse(
                channel_id
            )
            await self._on_ws_event(ws_id, effective_route, event)

        async def _on_unavailable() -> None:
            await self._stop_unavailable_stream(ws_id)

        await run_sse_stream(
            http_client=self._http_client,
            log_prefix="telegram",
            ws_id=ws_id,
            node_url_fn=self.router.get_node_url,
            token_factory=self._token_factory,
            on_event=_on_event,
            on_unavailable=_on_unavailable,
        )

    async def _stop_unavailable_stream(self, ws_id: str) -> None:
        """Retire this listener while retaining the route and session association."""
        task = self._sse_tasks.get(ws_id)
        if task is not None and task is not asyncio.current_task():
            return
        self._sse_tasks.pop(ws_id, None)
        self._clear_ws_state(ws_id)

    # -- event dispatch ------------------------------------------------------

    async def _on_ws_event(
        self,
        ws_id: str,
        route: TelegramRoute,
        event: ServerEvent,
    ) -> None:
        """Dispatch a typed server event to its per-event handler."""
        if isinstance(event, (ThinkingStartEvent, ThinkingStopEvent)):
            return

        if isinstance(event, ContentEvent):
            await self._handle_content(ws_id, route, event)
        elif isinstance(event, ApproveRequestEvent):
            await self._handle_approve_request(ws_id, route, event)
        elif isinstance(event, IntentVerdictEvent):
            await self._handle_intent_verdict(ws_id, event)
        elif isinstance(event, ApprovalResolvedEvent):
            await self._handle_approval_resolved(ws_id, event)
        elif isinstance(event, StreamEndEvent):
            await self._handle_stream_end(ws_id)
        elif isinstance(event, ErrorEvent):
            await self._handle_error(route, event)

    async def _handle_content(self, ws_id: str, route: TelegramRoute, event: ContentEvent) -> None:
        chat_id = int(route.chat) if route.chat else 0
        thread_id = route.thread_id or ""
        sm = self._streaming.get(ws_id)
        if sm is None or sm.chat_id != chat_id or sm.thread_id != thread_id:
            # Finalize the outgoing StreamingMessage before swapping so any
            # already-buffered tokens still land on the old thread instead
            # of being dropped (happens when the effective route flips
            # mid-stream — e.g. a user replies to a notification and pins
            # the conversation to a different chain).
            if sm is not None:
                await sm.finalize()
            thread = await self._thread_target(chat_id, thread_id)
            sm = StreamingMessage(
                bot=self._app.bot,
                chat_id=chat_id,
                thread_id=thread_id,
                thread=thread,
                max_length=self.config.max_message_length,
                edit_interval=self.config.streaming_edit_interval,
            )
            self._streaming[ws_id] = sm
        await sm.append(event.text)

    async def _handle_approve_request(
        self,
        ws_id: str,
        route: TelegramRoute,
        event: ApproveRequestEvent,
    ) -> None:
        chat_id = int(route.chat) if route.chat else 0
        thread_id = route.thread_id or ""
        owner_user_id = route.user_id

        # Every decision this handler takes is about THIS event's cycle —
        # forward its cycle_id so the resolution can't land on a sibling
        # round (parallel task agents can have several outstanding).
        cycle_id = event.cycle_id
        verdict = await self.router.evaluate_tool_policies(event.items)
        policy_handled = False
        if verdict.kind == "deny":
            denied = ", ".join(verdict.denied_tools)
            await self.router.send_approval(
                ws_id,
                cycle_id,
                approved=False,
                feedback=f"Blocked by tool policy: {denied}",
            )
            await self._reply(
                chat_id,
                f"<i>Tool blocked by admin policy: {_escape_html(denied)}</i>",
                thread_id=thread_id,
            )
            policy_handled = True
        elif verdict.kind == "allow":
            await self.router.send_approval(ws_id, cycle_id, approved=True)
            await self._reply(chat_id, "<i>Tool approved by policy.</i>", thread_id=thread_id)
            policy_handled = True

        if not policy_handled and (self.config.auto_approve or self._should_auto_approve(event)):
            await self.router.send_approval(ws_id, cycle_id, approved=True)
            await self._reply(chat_id, "<i>Tool auto-approved.</i>", thread_id=thread_id)
        elif not policy_handled:
            await self._send_approval_request(
                ws_id,
                cycle_id,
                event.items,
                chat_id,
                thread_id,
                owner_user_id,
            )

    async def _handle_intent_verdict(self, ws_id: str, event: IntentVerdictEvent) -> None:
        # Route the verdict onto the message whose cycle contains this
        # call_id — with several prompts live, editing "the" entry would
        # stack the judge section on the wrong message.  Legacy entries
        # (empty call_ids) accept any verdict for the ws, as before.
        entry = next(
            (
                v
                for (wid, _), v in self._pending_approval.items()
                if wid == ws_id and (not v.call_ids or event.call_id in v.call_ids)
            ),
            None,
        )
        if entry is None:
            return

        # Judge output is model-authored — escape each field before it is
        # interpolated into the bot's HTML framing.
        risk = _escape_html((event.risk_level or "medium").upper())
        confidence = _escape_html(str(event.confidence) if event.confidence else "N/A")
        verdict_html = (
            f"<b>Judge Verdict:</b> <code>{_escape_html(_code_name(event.func_name or 'tool'))}</code>\n"
            f"Risk: {risk} | Confidence: {confidence}"
        )
        summary = " ".join((event.intent_summary or "").splitlines()).strip()
        if summary:
            verdict_html += f"\nSummary:\n<pre>{_escape_html(summary[:1200])}</pre>"

        updated = _PendingApproval(
            chat_id=entry.chat_id,
            message_id=entry.message_id,
            owner_user_id=entry.owner_user_id,
            body_html=entry.body_html,
            cycle_id=entry.cycle_id,
            call_ids=entry.call_ids,
            verdicts=(*entry.verdicts, verdict_html),
        )
        self._pending_approval[(ws_id, entry.cycle_id)] = updated
        try:
            await self._app.bot.edit_message_text(
                chat_id=entry.chat_id,
                message_id=entry.message_id,
                text=entry.body_html + "\n\n" + "\n\n".join(updated.verdicts),
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            log.debug("telegram.verdict_message_edit_failed", ws_id=ws_id, exc_info=True)

    async def _handle_approval_resolved(self, ws_id: str, event: ApprovalResolvedEvent) -> None:
        # Route by the event's cycle_id; a pre-multi-cycle server (no
        # cycle_id) clears the ws's single tracked entry, as before.
        entry = pop_cycle_entry(self._pending_approval, ws_id, event.cycle_id)
        if entry is None:
            return

        label = "✅ Approved" if event.approved else "⛔ Denied"
        try:
            await self._app.bot.edit_message_text(
                chat_id=entry.chat_id,
                message_id=entry.message_id,
                text=entry.body_html + f"\n{label}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            log.debug("telegram.approval_resolved_edit_failed", ws_id=ws_id, exc_info=True)

    async def _handle_stream_end(self, ws_id: str) -> None:
        sm = self._streaming.pop(ws_id, None)
        if sm is not None:
            await sm.finalize()

        # Pop the notification-reply override so subsequent turns on this ws
        # default back to the session route.  Without the pop, one notification
        # reply pins all future responses to the notification thread until the
        # bot restarts.
        reply_route = self._notify_reply_routes.pop(ws_id, None)
        if (
            reply_route is not None
            and sm is not None
            and sm.message_id is not None
            and reply_route.chat
            and reply_route.user_id
        ):
            self._track_notification(sm.message_id, ws_id, reply_route)

        self._pop_ws_approvals(ws_id)

    async def _handle_error(self, route: TelegramRoute, event: ErrorEvent) -> None:
        # Error text can embed user/model-authored fragments (tool output,
        # provider messages) — escape the field before interpolation.
        safe_msg = _escape_html(event.message[:500]) if event.message else "An error occurred"
        try:
            await self._reply(
                int(route.chat) if route.chat else 0,
                f"⚠️ <b>Error:</b> {safe_msg}",
                thread_id=route.thread_id or "",
            )
        except Exception:
            log.debug("telegram.error_delivery_failed", exc_info=True)

    async def _send_approval_request(
        self,
        ws_id: str,
        correlation_id: str,
        items: list[dict[str, Any]],
        chat_id: int,
        thread_id: str,
        owner_user_id: str | None,
    ) -> None:
        tool_lines: list[str] = []
        body_len = 0
        truncated_items = 0
        for item in items:
            name = _escape_html(
                _code_name(item.get("approval_label") or item.get("func_name") or "tool")
            )
            # A call the person decides on (awaiting approval, not already refused) shows its
            # header when it has no preview, as a one-line bash command does.  Other items show only
            # their name, as before, so no header of theirs can crowd it off the card.
            raw_preview = item.get("preview", "")
            if not raw_preview and item.get("needs_approval") and not item.get("error"):
                raw_preview = item.get("header", "")
            preview = _escape_html(raw_preview[:_APPROVAL_PER_ITEM_PREVIEW]) if raw_preview else ""

            line = f"• <code>{name}</code>"
            if preview:
                line += f"\n<pre>{preview}</pre>"
            # +1 for the join newline
            if body_len + len(line) + 1 > _APPROVAL_TEXT_BUDGET:
                truncated_items = len(items) - len(tool_lines)
                break
            tool_lines.append(line)
            body_len += len(line) + 1

        if truncated_items > 0:
            tool_lines.append(f"<i>…and {truncated_items} more (preview truncated)</i>")

        marker = f"ts:{ws_id}|{correlation_id}|{owner_user_id or ''}"
        body_html = "<b>Tool Approval Required</b>\n" + "\n".join(tool_lines) + f"\n<i>{marker}</i>"

        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("Approve", callback_data="ts_approve"),
                    InlineKeyboardButton("Deny", callback_data="ts_deny"),
                ]
            ]
        )

        thread = await self._thread_target(chat_id, thread_id)
        try:
            message = await _send_html(
                self._app.bot,
                chat_id,
                body_html,
                thread,
                reply_markup=keyboard,
                max_length=self.config.max_message_length,
            )
        except Exception:
            log.warning("telegram.approval_post_failed", ws_id=ws_id, exc_info=True)
            return
        self._pending_approval[(ws_id, correlation_id)] = _PendingApproval(
            chat_id=chat_id,
            message_id=message.message_id,
            owner_user_id=owner_user_id,
            body_html=body_html,
            cycle_id=correlation_id,
            call_ids=frozenset(str(it.get("call_id", "")) for it in items if it.get("call_id")),
        )
        log.info(
            "telegram.pending_approval_stored",
            ws_id=ws_id,
            cycle_id=correlation_id,
            chat=chat_id,
            thread_id=thread_id,
            owner_user_id=owner_user_id,
        )

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an approve or deny button click from an approval prompt."""
        query = update.callback_query
        if query is None:
            return

        approved = query.data == "ts_approve"
        if not approved and query.data != "ts_deny":
            await query.answer()
            return

        message = query.message
        # Callback queries for messages older than 48 hours carry an
        # inaccessible message with no text — the marker is unreadable.
        parsed = _parse_approval_marker(getattr(message, "text", None))
        if parsed is None or message is None:
            await query.answer(
                "This approval can no longer be verified. Please retry from the active session.",
                show_alert=True,
            )
            return

        ws_id, correlation_id, owner_id = parsed
        verb = "approve" if approved else "deny"
        entry = get_cycle_entry(self._pending_approval, ws_id, correlation_id)

        log.info(
            "telegram.approval_actor_check",
            ws_id=ws_id,
            actor_user_id=query.from_user.id,
            owner_user_id=entry.owner_user_id if entry else None,
            has_entry=entry is not None,
        )

        if entry is None or not entry.owner_user_id:
            log.warning(
                "telegram.approval_missing_owner",
                ws_id=ws_id,
                actor_user_id=query.from_user.id,
            )
            await query.answer(
                "This approval can no longer be verified. Please retry from the active session.",
                show_alert=True,
            )
            return

        if str(query.from_user.id) != entry.owner_user_id:
            await query.answer(
                f"Only the session owner can {verb} this tool call.", show_alert=True
            )
            return

        await self.router.send_approval(ws_id, entry.cycle_id, approved=approved)
        # Drop the pending entry now that we've handled it locally. Otherwise
        # the subsequent ApprovalResolvedEvent will rewrite the message a
        # second time — a wasted edit and a visible flicker.
        self._pending_approval.pop((ws_id, entry.cycle_id), None)

        label = "✅ <b>Approved</b>" if approved else "⛔ <b>Denied</b>"
        try:
            await self._app.bot.edit_message_text(
                chat_id=entry.chat_id,
                message_id=entry.message_id,
                text=entry.body_html + f"\n{label}",
                parse_mode=ParseMode.HTML,
            )
        except Exception:
            log.debug(
                "telegram.approval_message_update_failed",
                ws_id=ws_id,
                event_key="approve" if approved else "deny",
                exc_info=True,
            )
        await query.answer("Approved" if approved else "Denied")
        log.info(
            "telegram.approval_response",
            ws_id=ws_id,
            correlation_id=correlation_id,
            approved=approved,
        )

    def _should_auto_approve(self, event: ApproveRequestEvent) -> bool:
        allowed = self.config.auto_approve_tools
        if not allowed or not event.items:
            return False

        for item in event.items:
            name = (
                item.get("func_name")
                or item.get("approval_label")
                or item.get("function", {}).get("name", "")
            )
            if name not in allowed:
                return False

        return True

    # -- notification plumbing -----------------------------------------------

    def _track_notification(
        self,
        message_id: int,
        ws_id: str,
        route: TelegramRoute,
    ) -> None:
        while len(self._notify_ws_map) >= self._MAX_NOTIFY_TRACKING:
            del self._notify_ws_map[next(iter(self._notify_ws_map))]
        chat_key = int(route.chat) if route.chat else 0
        self._notify_ws_map[(chat_key, message_id)] = (ws_id, route)
        log.info(
            "telegram.notification_tracked",
            message_id=message_id,
            ws_id=ws_id,
            chat=route.chat,
            user=route.user_id,
        )

    def _clear_notification_tracking_for_ws(self, ws_id: str) -> None:
        self._notify_reply_routes.pop(ws_id, None)
        stale = [key for key, entry in self._notify_ws_map.items() if entry[0] == ws_id]
        for key in stale:
            del self._notify_ws_map[key]

    def _track_session_msg(self, session_key: tuple[str, str], message_id: int) -> None:
        """Record a message ID as part of a reply-chain session."""
        tracked = self._session_msgs.get(session_key)
        if tracked is None:
            tracked = self._session_msgs[session_key] = OrderedDict()
        tracked[message_id] = None
        while len(tracked) > _SESSION_MSG_CAP:
            tracked.popitem(last=False)

    # -- forum cache ---------------------------------------------------------

    async def _thread_target(self, chat_id: int, thread_id: str) -> tuple[int, int]:
        """Send placement ``(message_thread_id, reply_to_message_id)``.

        Forum topics address by ``message_thread_id``; regular-group
        sessions reply to their chain anchor; DMs send plain (``(0, 0)``).
        """
        if not thread_id:
            return (0, 0)
        is_forum = self._forum_cache.get(chat_id)
        if is_forum is None:
            with contextlib.suppress(Exception):
                chat = await self._app.bot.get_chat(chat_id)
                is_forum = bool(getattr(chat, "is_forum", False))
            self._forum_cache[chat_id] = bool(is_forum)
        if is_forum:
            return (int(thread_id), 0)
        return (0, int(thread_id))

    async def _reply(self, chat_id: int, text: str, *, thread_id: str = "") -> None:
        """Post *text* into *chat_id*, honouring a thread anchor when present."""
        thread = await self._thread_target(chat_id, thread_id) if thread_id else (0, 0)
        await _send_html(self._app.bot, chat_id, text, thread)

    # -- ChannelAdapter protocol ---------------------------------------------

    async def send(self, channel_id: str, content: str) -> str:
        route = TelegramRoute.parse(channel_id)
        if not route.chat:
            raise ValueError(f"Telegram channel_id {channel_id!r} has no chat component")

        # Notification bodies are server/model-authored text (titles, task
        # names) with no bot-composed markup — the whole content is one
        # untrusted field.  Escape before chunking so tags and mentions
        # can't resolve on the wire.
        escaped = _escape_html(content)
        chat_id = int(route.chat)
        thread = await self._thread_target(chat_id, route.thread_id or "")

        last_id = ""
        for chunk in chunk_message(escaped, self.config.max_message_length - _SEND_LIMIT_SLACK):
            message = await _send_html(
                self._app.bot,
                chat_id,
                _render_escaped_html(chunk),
                thread,
                max_length=self.config.max_message_length,
            )
            last_id = str(message.message_id)

        return last_id

    async def send_notification(self, channel_id: str, content: str, ws_id: str) -> str:
        """Send a notification and track the reply mapping.

        *channel_id* is either a plain Telegram user ID (username
        targeting — the DM chat with that user) or a ``TelegramRoute``
        codec string for direct targeting.  DMs to users who never
        started a chat with the bot fail: Telegram delivers bot
        messages only inside chats the user opened.
        """
        route = TelegramRoute.parse(channel_id)
        if not route.chat:
            raise ValueError(f"Telegram channel_id {channel_id!r} has no chat component")
        if not route.user_id and not route.chat.lstrip("-").isdigit():
            # A non-numeric bare chat value is neither a user ID nor a
            # chat ID — reject before sending anywhere.
            raise ValueError(f"Telegram channel_id {channel_id!r} is not a valid target")

        message_id = await self.send(channel_id, content)
        if message_id and ws_id and route.user_id:
            self._track_notification(
                int(message_id),
                ws_id,
                TelegramRoute(chat=route.chat, user_id=route.user_id, thread_id=route.thread_id),
            )
        return message_id
