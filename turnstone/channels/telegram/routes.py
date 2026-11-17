"""Telegram channel-routing key dataclass.

A :class:`TelegramRoute` is the triple that uniquely identifies where a
Telegram conversation lives: ``(chat, user_id?, thread_id?)``.  It
round-trips through a ``chat:user_id:thread_id`` string so it can be used
as the opaque ``channel_id`` value stored in ``channel_routes``.

``thread_id`` carries the forum-topic ``message_thread_id`` in forum
supergroups, or the opener message ID that anchors a reply-chain session
in regular groups.  Telegram IDs never contain ``:`` so the codec is
unambiguous even though chat IDs may be negative.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelegramRoute:
    """Routing key for a Telegram conversation (chat / DM / topic or chain).

    ``to_channel_id`` and ``parse`` are inverses only for values the
    parser is willing to emit:

    - ``TelegramRoute("C")`` ↔ ``"C"``
    - ``TelegramRoute("C", "U")`` ↔ ``"C:U"``
    - ``TelegramRoute("C", "U", "T")`` ↔ ``"C:U:T"``

    Lax cases (not produced by ``to_channel_id`` but accepted by
    ``parse``):

    - Trailing colons drop to ``None`` — ``"C:"`` → ``TelegramRoute("C")``.
    - Extra colons fold into ``thread_id`` via ``split(":", 2)``, so
      ``"C:U:T:extra"`` → ``thread_id="T:extra"``.  No Telegram chat,
      user, or message ID contains ``:`` so this is safe in practice.
    """

    chat: str
    user_id: str | None = None
    thread_id: str | None = None

    @classmethod
    def parse(cls, channel_id: str) -> TelegramRoute:
        parts = channel_id.split(":", 2)
        chat = parts[0] if parts else ""
        user_id = parts[1] if len(parts) >= 2 and parts[1] else None
        thread_id = parts[2] if len(parts) == 3 and parts[2] else None
        return cls(chat=chat, user_id=user_id, thread_id=thread_id)

    def to_channel_id(self) -> str:
        if self.thread_id:
            return f"{self.chat}:{self.user_id or ''}:{self.thread_id}"
        if self.user_id:
            return f"{self.chat}:{self.user_id}"
        return self.chat

    @property
    def has_user(self) -> bool:
        return bool(self.user_id)

    @property
    def has_thread(self) -> bool:
        return bool(self.thread_id)
