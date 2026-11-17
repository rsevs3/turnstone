"""Telegram-specific configuration."""

from __future__ import annotations

from dataclasses import dataclass, field

from turnstone.channels._config import ChannelConfig


@dataclass
class TelegramConfig(ChannelConfig):
    """Configuration for the Telegram channel adapter.

    Uses long polling so no public URL or webhook endpoint is required —
    the bot connects outbound to Telegram's Bot API.

    Requires one token:
    - bot_token: issued by @BotFather when the bot is created

    To create it:
    1. Open @BotFather in Telegram and run ``/newbot``
    2. Copy the token it issues
    3. For group support, run ``/setprivacy`` → select the bot → **Disable**,
       so the bot receives non-command messages in groups (DMs are unaffected)
    4. For forum supergroups, grant the bot the **Manage Topics** admin right
       so it can create a topic per session
    """

    bot_token: str = ""  # BotFather token
    allowed_chats: list[int] = field(default_factory=list)  # empty = all
    max_message_length: int = 4096  # Telegram's own message limit
    streaming_edit_interval: float = 1.5  # seconds between message edits
