"""Minimal ntfy.sh (or self-hosted ntfy) push-notification client.

Uses only the stdlib (urllib) so notifications don't need a new dependency. Network failures
are caught and logged rather than raised -- a missed push notification should never crash an
hours-long EA run.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

NTFY_TOPIC_ENV_VAR = "NTFY_TOPIC"
NTFY_SERVER_ENV_VAR = "NTFY_SERVER"
DEFAULT_NTFY_SERVER = "https://ntfy.sh"
DEFAULT_NTFY_TOPIC = "avlnn-pipeline"   # ntfy.sh topics are public -- pick your own unguessable
                                         # value via --ntfy-topic/NTFY_TOPIC rather than using
                                         # this default for anything you actually subscribe to


@dataclasses.dataclass(frozen=True)
class NtfyConfig:
    topic: str
    server: str = DEFAULT_NTFY_SERVER

    @classmethod
    def from_env(cls) -> "NtfyConfig | None":
        topic = os.environ.get(NTFY_TOPIC_ENV_VAR)
        if not topic:
            return None
        return cls(topic=topic, server=os.environ.get(NTFY_SERVER_ENV_VAR, DEFAULT_NTFY_SERVER))


def resolve_ntfy_config(
    topic_arg: str | None = None, server_arg: str | None = None,
) -> "NtfyConfig | None":
    """Shared CLI resolution: explicit argument (including '' to disable) beats the
    NTFY_TOPIC/NTFY_SERVER env vars, which beat the project defaults."""
    if topic_arg is not None:
        topic = topic_arg
    else:
        topic = os.environ.get(NTFY_TOPIC_ENV_VAR, DEFAULT_NTFY_TOPIC)
    server = server_arg or os.environ.get(NTFY_SERVER_ENV_VAR, DEFAULT_NTFY_SERVER)
    return NtfyConfig(topic=topic, server=server) if topic else None


def send_ntfy(
    config: NtfyConfig,
    message: str,
    title: str | None = None,
    priority: str | None = None,
    tags: list[str] | None = None,
    timeout_s: float = 5.0,
) -> bool:
    """POSTs a notification to the given ntfy topic. Returns True on success; on any failure
    it logs a warning and returns False rather than raising, so a flaky network never takes
    down the EA run that's calling this."""
    url = f"{config.server.rstrip('/')}/{config.topic}"
    headers = {}
    if title:
        headers["Title"] = title
    if priority:
        headers["Priority"] = priority
    if tags:
        headers["Tags"] = ",".join(tags)

    req = urllib.request.Request(url, data=message.encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except Exception:
        logger.warning("ntfy notification to %s failed", url, exc_info=True)
        return False
