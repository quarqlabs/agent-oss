"""
Quarq local control console.

This file is intentionally separate from agent.py. It starts the existing
FastAPI worker and connects local CLI input to /api/chat. Channel integrations
such as Telegram live in main.py as API webhooks.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv
from rich.panel import Panel
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.rule import Rule
from rich.text import Text

from agent_config import load_agent_config
from agent_tools_config import format_slug_list, load_enabled_cloud_tools
from coding_agents.config import coding_config_summary, load_settings
from coding_agents.task_store import CodingTaskStore

try:
    from textual.app import App as TextualApp
    from textual.binding import Binding
    from textual.widgets import RichLog, Static
    from textual.widgets import TextArea as TextualTextArea
except ImportError:  # pragma: no cover - supported fallback before deps install
    Binding = None
    RichLog = None
    Static = None
    TextualApp = None
    TextualTextArea = None

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.layout import Dimension, HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.margins import ScrollbarMargin
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.styles import Style as PromptStyle
    from prompt_toolkit.widgets import TextArea
except ImportError:  # pragma: no cover - supported fallback before deps install
    Application = None
    Dimension = None
    FormattedTextControl = None
    HSplit = None
    KeyBindings = None
    Keys = None
    Layout = None
    MouseEventType = None
    PromptStyle = None
    ScrollbarMargin = None
    TextArea = None
    Window = None


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
LAUNCH_CWD = Path.cwd().resolve()
os.environ.setdefault("QUARQ_LAUNCH_CWD", str(LAUNCH_CWD))
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
TELEGRAM_WEBHOOK_PATH = "/api/telegram/webhook"
EVENT_POLL_INTERVAL = 0.5
JOB_POLL_INTERVAL = 3.0
TELEGRAM_API_BASE = "https://api.telegram.org"
TRYCLOUDFLARE_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
TELEGRAM_SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
CLI_CONFIG_AGENT_ID = re.sub(r"[^a-zA-Z0-9_.-]", "_", os.getenv("AGENT_ID", "local_agent"))
CLI_CONFIG_PATH = BASE_DIR / "local_memory" / CLI_CONFIG_AGENT_ID / "agent_cli.json"

for noisy_logger in ("httpx", "httpcore", "openai", "openai._base_client"):
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)


if all([Binding, RichLog, Static, TextualApp, TextualTextArea]):

    class QuarqTextualApp(TextualApp):
        CSS = """
        Screen {
            background: #111318;
            color: #c9d1d9;
        }

        Screen > .screen--selection,
        .screen--selection {
            background: #1f6feb;
            color: #ffffff;
            text-style: bold;
        }

        #header {
            height: 8;
            padding: 1 2 0 2;
            background: #111318;
            color: #8b949e;
        }

        #transcript {
            height: 1fr;
            background: #111318;
            color: #c9d1d9;
            overflow-x: hidden;
            overflow-y: scroll;
            scrollbar-background: #20242c;
            scrollbar-color: #60a5fa;
            scrollbar-color-hover: #93c5fd;
            scrollbar-size-horizontal: 0;
            border-bottom: solid #3b414a;
        }

        #input_label {
            height: 1;
            padding: 0 2;
            background: #111318;
            color: #aeb6c2;
        }

        #command_palette {
            height: auto;
            max-height: 9;
            margin: 0 2;
            padding: 0 1;
            background: #111318;
            color: #8b949e;
            border-top: solid #3b414a;
        }

        #message {
            height: 4;
            margin: 0 2;
            background: #2f3540;
            color: #f8fafc;
            border: none;
            padding: 0 1;
        }

        #footer {
            height: 1;
            padding: 0 2;
            background: #111318;
            color: #8b949e;
        }
        """

        BINDINGS = [
            Binding("enter", "send_message", "Send", priority=True),
            Binding("shift+enter", "insert_newline", "New line", priority=True),
            Binding("ctrl+enter", "send_message", "Send", priority=True),
            Binding("escape,enter", "send_message", "Send", priority=True),
            Binding("ctrl+s", "send_message", "Send", show=False, priority=True),
            Binding("tab", "complete_command", "Complete", show=False, priority=True),
            Binding("ctrl+n", "next_suggestion", "Next suggestion", show=False, priority=True),
            Binding("ctrl+p", "previous_suggestion", "Previous suggestion", show=False, priority=True),
            Binding("super+a", "select_message_all", "Select input", show=False, priority=True),
            Binding("super+c", "copy_active_selection", "Copy selection", show=False, priority=True),
            Binding("pageup", "page_up", "Older"),
            Binding("pagedown", "page_down", "Newer"),
            Binding("ctrl+up", "line_up", "Older"),
            Binding("ctrl+down", "line_down", "Newer"),
            Binding("alt+k", "line_up", "Older"),
            Binding("alt+j", "line_down", "Newer"),
            Binding("ctrl+c", "quit_requested", "Quit", priority=True),
            Binding("ctrl+d", "quit_requested", "Quit", priority=True),
            Binding("ctrl+q", "quit_requested", "Quit", priority=True),
        ]

        def __init__(self, ui: "TextualTerminalUi"):
            super().__init__()
            self.ui = ui

        def compose(self) -> Any:
            yield Static(self.ui.header_renderable(), id="header")
            yield RichLog(
                id="transcript",
                min_width=1,
                wrap=True,
                highlight=False,
                markup=False,
            )
            yield Static("", id="command_palette")
            yield Static("Message Quarq   Enter = send   Shift+Enter = newline", id="input_label")
            yield TextualTextArea(
                soft_wrap=True,
                show_line_numbers=False,
                compact=True,
                placeholder="Describe a task or ask Quarq...",
                id="message",
            )
            yield Static("", id="footer")

        def on_mount(self) -> None:
            self.refresh_header()
            self.refresh_transcript()
            self.refresh_footer()
            self.refresh_command_palette()
            self.query_one("#message", TextualTextArea).focus()

        def action_quit_requested(self) -> None:
            self.ui.input_queue.put_nowait("/quit")

        def action_select_message_all(self) -> None:
            message = self.query_one("#message", TextualTextArea)
            try:
                self.screen.clear_selection()
            except Exception:
                pass
            message.focus()
            message.action_select_all()

        def action_copy_active_selection(self) -> None:
            message = self.query_one("#message", TextualTextArea)
            selected_message_text = str(getattr(message, "selected_text", "") or "")
            if selected_message_text:
                self.copy_to_clipboard(selected_message_text)
                self.ui.set_status("copied input selection", "ready")
                self.refresh_footer()
                return

            selected_screen_text = ""
            try:
                selected_screen_text = str(self.screen.get_selected_text() or "")
            except Exception:
                selected_screen_text = ""

            if not selected_screen_text.strip():
                self.ui.set_status("select text first", "notice")
                self.refresh_footer()
                return

            if len(selected_screen_text) > 12000 or selected_screen_text.count("\n") > 200:
                self.ui.set_status("selection too large", "notice")
                self.refresh_footer()
                return

            self.copy_to_clipboard(selected_screen_text)
            self.ui.set_status("copied selection", "ready")
            self.refresh_footer()

        def action_send_message(self) -> None:
            message = self.query_one("#message", TextualTextArea)
            text = str(message.text or "").strip()
            if not text:
                return
            message.clear()
            self.refresh_command_palette()
            self.ui.input_queue.put_nowait(text)

        def action_insert_newline(self) -> None:
            message = self.query_one("#message", TextualTextArea)
            message.insert("\n")
            self.refresh_command_palette()

        def action_complete_command(self) -> None:
            message = self.query_one("#message", TextualTextArea)
            suggestions = command_suggestions(str(message.text or ""))
            if not suggestions:
                return

            index = min(max(self.ui.suggestion_index, 0), len(suggestions) - 1)
            message.load_text(suggestions[index]["insert"])
            try:
                message.move_cursor(message.document.end)
            except Exception:
                pass
            self.refresh_command_palette()

        def action_next_suggestion(self) -> None:
            suggestions = command_suggestions(str(self.query_one("#message", TextualTextArea).text or ""))
            if not suggestions:
                return
            self.ui.suggestion_index = (self.ui.suggestion_index + 1) % len(suggestions)
            self.refresh_command_palette()

        def action_previous_suggestion(self) -> None:
            suggestions = command_suggestions(str(self.query_one("#message", TextualTextArea).text or ""))
            if not suggestions:
                return
            self.ui.suggestion_index = (self.ui.suggestion_index - 1) % len(suggestions)
            self.refresh_command_palette()

        def action_page_up(self) -> None:
            self.scroll_transcript("page_up")

        def action_page_down(self) -> None:
            self.scroll_transcript("page_down")

        def action_line_up(self) -> None:
            self.scroll_transcript("up")

        def action_line_down(self) -> None:
            self.scroll_transcript("down")

        def refresh_header(self) -> None:
            header = self.query_one("#header", Static)
            header.update(self.ui.header_renderable())

        def refresh_footer(self) -> None:
            footer = self.query_one("#footer", Static)
            footer.update(self.ui.footer_text())

        def refresh_command_palette(self) -> None:
            try:
                palette = self.query_one("#command_palette", Static)
                message = self.query_one("#message", TextualTextArea)
            except Exception:
                return
            suggestions = command_suggestions(str(message.text or ""))
            signature = tuple(option["insert"] for option in suggestions)
            if signature != self.ui.suggestion_signature:
                self.ui.suggestion_signature = signature
                self.ui.suggestion_index = 0
            if suggestions:
                self.ui.suggestion_index = min(self.ui.suggestion_index, len(suggestions) - 1)
            palette.display = bool(suggestions)
            if suggestions:
                palette.update(render_command_suggestions(suggestions, self.ui.suggestion_index))
            else:
                palette.update("")

        def on_text_area_changed(self, event: Any) -> None:
            if event.text_area.id == "message":
                self.refresh_command_palette()

        def on_resize(self) -> None:
            self.refresh_transcript()

        def refresh_transcript(self) -> None:
            log = self.query_one("#transcript", RichLog)
            log.show_horizontal_scrollbar = False
            log.scroll_x = 0
            log.clear()
            width = transcript_render_width(log)
            for block in self.ui.output_blocks:
                log.write(render_textual_block(block), width=width, shrink=True)
            log.show_horizontal_scrollbar = False
            log.scroll_x = 0
            if self.ui.follow_output:
                call_scroll_method(log, "scroll_end")

        def scroll_transcript(self, direction: str) -> None:
            log = self.query_one("#transcript", RichLog)
            methods = {
                "page_up": "scroll_page_up",
                "page_down": "scroll_page_down",
                "up": "scroll_up",
                "down": "scroll_down",
                "bottom": "scroll_end",
            }
            method_name = methods.get(direction)
            if method_name:
                call_scroll_method(log, method_name)
            self.ui.follow_output = direction in {"page_down", "down", "bottom"} and is_scroll_at_end(log)
            self.refresh_footer()


else:
    QuarqTextualApp = None


class TextualTerminalUi:
    def __init__(self, api_base: str):
        if QuarqTextualApp is None:
            raise RuntimeError("textual is not available")

        self.api_base = api_base
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.output_blocks: list[dict[str, Any]] = []
        self.max_blocks = 1000
        self.follow_output = True
        self.suggestion_index = 0
        self.suggestion_signature: tuple[str, ...] = ()
        self.status_text = "ready"
        self.status_kind = "ready"
        self.model_label = model_label()
        self.directory_label = compact_path(LAUNCH_CWD)
        self.agent_name = load_cli_agent_name()
        self.agent_version = os.getenv("QUARQ_AGENT_VERSION", "v0.4.4").strip() or "v0.4.4"
        self.coding_agent_label = "codex"
        self.coding_workspace_label = compact_path(LAUNCH_CWD)
        self.coding_network_label = "net:on"
        self.refresh_dynamic_labels()
        self.connected_channels: set[str] = set()
        self.default_start_channels = set(load_cli_config().get("startup_channels", []))
        self.output_blocks.append(welcome_block())
        self.app = QuarqTextualApp(self)

    def header_renderable(self) -> Panel:
        self.refresh_dynamic_labels()
        text = Text()
        text.append("›  ", style="#c9d1d9")
        text.append(self.agent_name, style="bold #f8fafc")
        text.append(f" ({self.agent_version})", style="#8b949e")
        text.append("\nmodel:     ", style="#8b949e")
        text.append(self.model_label, style="bold #f8fafc")
        text.append("\ndirectory: ", style="#8b949e")
        text.append(self.directory_label, style="bold #e5e7eb")
        text.append("\napi:       ", style="#8b949e")
        text.append(self.api_base, style="#c9d1d9")
        text.append("\nchannels:  ", style="#8b949e")
        text.append(channel_summary(self.connected_channels), style="bold #c9d1d9")
        text.append("    startup: ", style="#8b949e")
        text.append(channel_summary(self.default_start_channels), style="#c9d1d9")
        text.append("\ncoding:    ", style="#8b949e")
        text.append(self.coding_agent_label, style="bold #a78bfa")
        text.append(" · ", style="#6b7280")
        text.append(self.coding_workspace_label, style="#c9d1d9")
        text.append(" · ", style="#6b7280")
        text.append(self.coding_network_label, style="#c9d1d9")
        return Panel.fit(text, border_style="#5a606b", padding=(0, 1))

    def refresh_dynamic_labels(self) -> None:
        self.agent_name = load_cli_agent_name()
        coding_config = coding_config_summary()
        self.coding_agent_label = str(
            coding_config.get("default_provider_label")
            or coding_config.get("default_provider")
            or "codex"
        )
        self.coding_workspace_label = compact_path(Path(str(coding_config.get("workspace_root") or BASE_DIR)))
        self.coding_network_label = "net:on" if coding_config.get("network_access", True) else "net:off"

    def refresh_header(self) -> None:
        self._schedule(self.app.refresh_header)

    def set_channel_connected(self, channel_type: str) -> None:
        self.connected_channels.add(normalize_channel_type(channel_type))
        self._schedule(self.app.refresh_header)

    def set_default_start_channels(self, channels: list[str]) -> None:
        self.default_start_channels = set(channels)
        self._schedule(self.app.refresh_header)

    def footer_text(self) -> Text:
        text = Text()
        text.append(self.model_label, style="bold #f8fafc")
        text.append(" · ", style="#6b7280")
        text.append(self.directory_label, style="bold #9be9a8")
        text.append("    status ", style="#6b7280")
        text.append(self.status_text, style=status_style(self.status_kind))
        text.append("    copy: drag+Cmd+C", style="#8b949e")
        text.append("    quit: Ctrl+C", style="#8b949e")
        text.append("    scroll: wheel, PageUp/PageDown, Ctrl+Up/Ctrl+Down", style="#8b949e")
        return text

    async def run(self) -> None:
        await self.app.run_async()

    def close(self) -> None:
        if getattr(self.app, "is_running", False):
            self.app.exit()

    async def read_input(self) -> str:
        return await self.input_queue.get()

    def set_status(self, text: str, kind: str = "notice") -> None:
        self.status_text = text
        self.status_kind = normalize_status_kind(kind)
        self._schedule(self.app.refresh_footer)

    def output_page_size(self) -> int:
        rows = shutil.get_terminal_size((100, 32)).lines
        return max(5, rows - 7)

    def scroll_output(self, delta: int) -> None:
        if delta < 0:
            direction = "page_up" if abs(delta) >= self.output_page_size() else "up"
            self.follow_output = False
        else:
            direction = "page_down" if abs(delta) >= self.output_page_size() else "down"
        self._schedule(lambda: self.app.scroll_transcript(direction))

    def scroll_to_bottom(self) -> None:
        self.follow_output = True
        self._schedule(lambda: self.app.scroll_transcript("bottom"))

    def append(
        self,
        title: str,
        body: str = "",
        kind: str = "event",
        details: str = "",
        replace_key: str | None = None,
    ) -> None:
        block = {
            "key": replace_key,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "kind": normalize_event_kind(kind),
            "title": str(title),
            "body": str(body or ""),
            "details": str(details or ""),
        }

        replaced = False
        if replace_key:
            for index in range(len(self.output_blocks) - 1, -1, -1):
                if self.output_blocks[index].get("key") == replace_key:
                    self.output_blocks[index] = block
                    replaced = True
                    break

        if not replaced:
            self.output_blocks.append(block)

        self.output_blocks = self.output_blocks[-self.max_blocks :]
        self._schedule(self.app.refresh_transcript)

    def _schedule(self, callback: Any) -> None:
        if not getattr(self.app, "is_running", False):
            return
        try:
            self.app.call_later(callback)
        except Exception:
            try:
                callback()
            except Exception:
                pass


class TerminalUi:
    def __init__(self, api_base: str):
        if not all(
            [
                Application,
                Dimension,
                FormattedTextControl,
                HSplit,
                KeyBindings,
                Keys,
                Layout,
                MouseEventType,
                PromptStyle,
                ScrollbarMargin,
                TextArea,
                Window,
            ]
        ):
            raise RuntimeError("prompt_toolkit is not available")

        self.api_base = api_base
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.output_blocks: list[dict[str, Any]] = []
        self.max_blocks = 1000
        self.follow_output = True
        self.scroll_top_line = 0
        self.status_text = "ready"
        self.status_kind = "ready"
        self.output_control = FormattedTextControl(
            self._output_fragments,
            focusable=False,
            show_cursor=False,
        )
        self.output_control.mouse_handler = self._output_mouse_handler
        self.output = Window(
            self.output_control,
            wrap_lines=False,
            style="class:output",
            height=Dimension(weight=1),
            get_vertical_scroll=self._get_output_vertical_scroll,
            right_margins=[ScrollbarMargin(display_arrows=False)],
        )
        self.input = TextArea(
            multiline=False,
            height=1,
            prompt=[("class:input.prompt", "› ")],
            accept_handler=self._accept_input,
            wrap_lines=False,
            style="class:input.field",
        )

        key_bindings = KeyBindings()

        @key_bindings.add("c-c")
        @key_bindings.add("c-d")
        @key_bindings.add("c-q")
        def _(event: Any) -> None:
            self.input_queue.put_nowait("/quit")

        @key_bindings.add("pageup", eager=True, is_global=True)
        @key_bindings.add(Keys.ControlPageUp, eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_output(-self.output_page_size())

        @key_bindings.add("pagedown", eager=True, is_global=True)
        @key_bindings.add(Keys.ControlPageDown, eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_output(self.output_page_size())

        @key_bindings.add(Keys.ControlUp, eager=True, is_global=True)
        @key_bindings.add("escape", "k", eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_output(-3)

        @key_bindings.add(Keys.ControlDown, eager=True, is_global=True)
        @key_bindings.add("escape", "j", eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_output(3)

        @key_bindings.add(Keys.ScrollUp, eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_output(-3)

        @key_bindings.add(Keys.ScrollDown, eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_output(3)

        @key_bindings.add("escape", "v", eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_output(-self.output_page_size())

        @key_bindings.add("escape", ">", eager=True, is_global=True)
        def _(event: Any) -> None:
            self.scroll_to_bottom()

        root = HSplit(
            [
                Window(
                    FormattedTextControl(self._header_fragments),
                    height=6,
                    style="class:header",
                ),
                Window(char="─", height=1, style="class:rule"),
                self.output,
                Window(char="─", height=1, style="class:rule"),
                Window(
                    FormattedTextControl(self._input_label_fragments),
                    height=1,
                    style="class:input.label",
                ),
                self.input,
                Window(
                    FormattedTextControl(self._footer_fragments),
                    height=1,
                    style="class:footer",
                ),
            ],
            height=Dimension(weight=1),
            style="class:app",
        )
        style = PromptStyle.from_dict(
            {
                "app": "bg:#111318",
                "header": "bg:#111318 #e5e7eb",
                "header.title": "bold #f8fafc",
                "header.dim": "#8b949e",
                "rule": "#3b414a",
                "output": "bg:#111318 #c9d1d9",
                "event.time": "#6b7280",
                "event.notice": "#fbbf24",
                "event.ready": "#22c55e",
                "event.tunnel": "#c084fc",
                "event.user": "#60a5fa",
                "event.agent": "#2dd4bf",
                "event.coding": "#a78bfa",
                "event.error": "#f87171",
                "event.event": "#94a3b8",
                "event.title": "bold #e5e7eb",
                "event.detail.key": "#94a3b8",
                "event.detail.value": "#38bdf8",
                "event.detail.metric": "#a78bfa",
                "md": "#d1d5db",
                "md.dim": "#8b949e",
                "md.bold": "bold #f8fafc",
                "md.code": "bg:#272c35 #fbbf24",
                "md.codeblock": "bg:#20242c #c4b5fd",
                "md.heading": "bold #60a5fa",
                "md.list.marker": "#22d3ee",
                "md.url": "underline #2dd4bf",
                "separator": "#3b414a",
                "scrollbar.background": "bg:#20242c #303640",
                "scrollbar.button": "bg:#60a5fa #60a5fa",
                "scrollbar.end": "bg:#93c5fd #93c5fd",
                "input.label": "bg:#2f3540 #f8fafc",
                "input.label.dim": "bg:#2f3540 #aeb6c2",
                "input.field": "bg:#3b424d #f8fafc",
                "input.prompt": "bg:#3b424d bold #f8fafc",
                "footer": "bg:#111318 #8b949e",
                "footer.dim": "bg:#111318 #6b7280",
                "status.ready": "bg:#111318 #22c55e",
                "status.busy": "bg:#111318 #fbbf24",
                "status.error": "bg:#111318 #f87171",
                "status.notice": "bg:#111318 #60a5fa",
            }
        )
        self.app = Application(
            layout=Layout(root, focused_element=self.input),
            key_bindings=key_bindings,
            full_screen=True,
            mouse_support=True,
            style=style,
        )

    def _header_fragments(self) -> list[tuple[str, str]]:
        coding_config = coding_config_summary()
        coding_label = (
            coding_config.get("default_provider_label")
            or coding_config.get("default_provider")
            or "codex"
        )
        workspace_label = compact_path(Path(str(coding_config.get("workspace_root") or LAUNCH_CWD)))
        return [
            ("class:header.title", f" {load_cli_agent_name()}"),
            ("class:header.dim", f"\n directory {compact_path(LAUNCH_CWD)}"),
            ("class:header.dim", f"\n api {self.api_base}"),
            ("class:header.dim", "\n channels local"),
            ("class:header.dim", f"\n coding {coding_label} · {workspace_label}"),
            ("class:header.dim", "\n commands /help /status /tools /cloud-tools /coding /coding-network /connect /wipe /quit"),
        ]

    def _input_label_fragments(self) -> list[tuple[str, str]]:
        return [
            ("class:input.label", " Message Quarq "),
            ("class:input.label.dim", "Enter to send"),
        ]

    def _footer_fragments(self) -> list[tuple[str, str]]:
        return [
            ("class:footer.dim", " status "),
            (f"class:status.{self.status_kind}", self.status_text),
            ("class:footer.dim", f"   transcript {self.scroll_position_text()}"),
            ("class:footer.dim", "   quit: Ctrl+C   older: PageUp/Ctrl+Up/Alt+K   newer: PageDown/Ctrl+Down/Alt+J "),
        ]

    def _output_fragments(self) -> list[tuple[str, str]]:
        lines = self.output_lines()
        if self.follow_output:
            self.scroll_top_line = self.max_output_scroll()

        if not lines:
            return [("class:md.dim", "")]

        fragments: list[tuple[str, str]] = []
        for index, line in enumerate(lines):
            fragments.extend(line)
            if index < len(lines) - 1:
                fragments.append(("", "\n"))
        return fragments

    def _get_output_vertical_scroll(self, _window: Any) -> int:
        max_scroll = self.max_output_scroll()
        if self.follow_output:
            self.scroll_top_line = max_scroll
        self.scroll_top_line = max(0, min(self.scroll_top_line, max_scroll))
        return self.scroll_top_line

    def _output_mouse_handler(self, mouse_event: Any) -> Any:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.scroll_output(-3)
            return None

        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.scroll_output(3)
            return None

        return NotImplemented

    def _accept_input(self, buffer: Any) -> bool:
        text = buffer.text
        buffer.text = ""
        self.input_queue.put_nowait(text)
        return True

    async def run(self) -> None:
        await self.app.run_async()

    def close(self) -> None:
        if self.app.is_running:
            self.app.exit()

    async def read_input(self) -> str:
        return await self.input_queue.get()

    def set_status(self, text: str, kind: str = "notice") -> None:
        self.status_text = text
        self.status_kind = normalize_status_kind(kind)
        self.app.invalidate()

    def output_page_size(self) -> int:
        rows = shutil.get_terminal_size((100, 32)).lines
        fixed_rows = 4 + 1 + 1 + 1 + 1
        return max(5, rows - fixed_rows - 2)

    def output_lines(self) -> list[list[tuple[str, str]]]:
        fragments: list[tuple[str, str]] = []
        for block in self.output_blocks:
            fragments.extend(block["fragments"])
        return wrap_fragment_lines(
            split_fragments_into_lines(fragments),
            self.output_text_width(),
        )

    def output_text_width(self) -> int:
        columns = shutil.get_terminal_size((100, 32)).columns
        return max(24, columns - 3)

    def max_output_scroll(self) -> int:
        return max(0, len(self.output_lines()) - self.output_page_size())

    def scroll_output(self, delta: int) -> None:
        previous = self.scroll_top_line
        self.scroll_top_line = max(0, min(self.max_output_scroll(), self.scroll_top_line + delta))
        self.follow_output = self.scroll_top_line >= self.max_output_scroll()
        if delta > 0 and self.scroll_top_line == previous and self.follow_output:
            self.status_text = "already at newest"
            self.status_kind = "notice"
        elif delta < 0 and self.scroll_top_line == previous and self.scroll_top_line == 0:
            self.status_text = "already at oldest"
            self.status_kind = "notice"
        self.app.invalidate()

    def scroll_to_bottom(self) -> None:
        self.scroll_top_line = self.max_output_scroll()
        self.follow_output = True
        self.app.invalidate()

    def scroll_position_text(self) -> str:
        total = len(self.output_lines())
        if total == 0:
            return "0/0"

        page_size = self.output_page_size()
        if total <= page_size:
            return f"all {total}"

        start = min(self.scroll_top_line + 1, total)
        end = min(self.scroll_top_line + page_size, total)
        suffix = " bottom" if self.follow_output else ""
        return f"{start}-{end}/{total}{suffix}"

    def append(
        self,
        title: str,
        body: str = "",
        kind: str = "event",
        details: str = "",
        replace_key: str | None = None,
    ) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        normalized_kind = normalize_event_kind(kind)
        block_fragments: list[tuple[str, str]] = [
            ("class:event.time", timestamp),
            ("", "  "),
            (f"class:event.{normalized_kind}", f"{normalized_kind:<6}"),
            ("", "  "),
            ("class:event.title", title),
        ]
        if details:
            block_fragments.append(("", "  "))
            block_fragments.extend(render_details(details))
        block_fragments.append(("", "\n"))

        rendered_body = render_markdown_body(str(body or ""))
        if rendered_body:
            block_fragments.extend(rendered_body)
            if block_fragments[-1][1] != "\n":
                block_fragments.append(("", "\n"))

        block_fragments.append(("class:separator", separator_text() + "\n"))
        block = {"key": replace_key, "fragments": block_fragments}

        replaced = False
        if replace_key:
            for index in range(len(self.output_blocks) - 1, -1, -1):
                if self.output_blocks[index].get("key") == replace_key:
                    self.output_blocks[index] = block
                    replaced = True
                    break

        if not replaced:
            self.output_blocks.append(block)

        self.output_blocks = self.output_blocks[-self.max_blocks :]
        if self.follow_output:
            self.scroll_top_line = self.max_output_scroll()
        self.app.invalidate()


class EventLog:
    def __init__(self, console: Console, ui: TerminalUi | None = None):
        self.console = console
        self.ui = ui
        self._lock = asyncio.Lock()

    async def emit(self, title: str, body: str = "", style: str = "white") -> None:
        async with self._lock:
            if self.ui is not None:
                self.ui.append(title, body, kind=event_kind_from_style(style))
                return

            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console.print(
                Text.assemble(
                    ("  ", "dim"),
                    (timestamp, "dim"),
                    ("  ", "dim"),
                    (title, f"bold {style}"),
                )
            )
            if body:
                self.console.print(Padding(Text(body, style="dim"), (0, 0, 0, 2)))

    async def emit_api_event(self, event: dict[str, Any]) -> None:
        kind = str(event.get("kind") or "event")
        style = {
            "request": "bright_black",
            "response": "white",
            "telegram": "bright_black",
            "system": "bright_black",
            "job": "yellow",
            "coding": "magenta",
            "warning": "yellow",
            "error": "red",
        }.get(kind, "white")

        title = str(event.get("title") or kind)
        timestamp = str(event.get("time") or "")[11:19] or datetime.now().strftime("%H:%M:%S")
        details = format_event_details(event)
        message = str(event.get("message") or "").strip()
        replace_key = progress_replace_key(event)

        async with self._lock:
            if self.ui is not None:
                ui_kind = "coding" if kind == "coding" else event_prefix(kind).strip()
                self.ui.append(
                    title,
                    message,
                    kind=ui_kind,
                    details=details,
                    replace_key=replace_key,
                )
                return

            self.console.print(
                Text.assemble(
                    ("  ", "dim"),
                    (timestamp, "dim"),
                    ("  ", "dim"),
                    (event_prefix(kind), style),
                    (title, f"bold {style}"),
                    (f"  {details}", "dim") if details else "",
                )
            )
            if message:
                self.console.print(Padding(Text(message, style="white"), (0, 0, 0, 2)))


class QuarqApiClient:
    def __init__(self, base_url: str, events: EventLog):
        self.base_url = base_url.rstrip("/")
        self.events = events
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=None)

    async def close(self) -> None:
        await self.client.aclose()

    async def health(self) -> dict[str, Any]:
        response = await self.client.get("/")
        response.raise_for_status()
        return response.json()

    async def wipe_memories(self) -> dict[str, Any]:
        response = await self.client.post("/api/memories/wipe")
        response.raise_for_status()
        return response.json()

    async def chat(
        self,
        prompt: str,
        channel_type: str,
        current_date: str | None = None,
        skip_learning: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "channel_type": channel_type,
            "skip_learning": skip_learning,
            "current_date": current_date,
        }
        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        return response.json()

    async def create_chat_job(
        self,
        prompt: str,
        channel_type: str,
        current_date: str | None = None,
        skip_learning: bool = False,
    ) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "channel_type": channel_type,
            "skip_learning": skip_learning,
            "current_date": current_date,
        }
        response = await self.client.post("/api/jobs", json=payload)
        response.raise_for_status()
        return response.json().get("job", {})

    async def get_job(self, job_id: str) -> dict[str, Any]:
        response = await self.client.get(f"/api/jobs/{job_id}")
        response.raise_for_status()
        return response.json().get("job", {})

    async def events_since(self, after: int) -> list[dict[str, Any]]:
        response = await self.client.get("/api/events", params={"after": after})
        response.raise_for_status()
        return response.json().get("events", [])

    async def list_coding_agents(self) -> dict[str, Any]:
        response = await self.client.get("/api/coding-agents")
        response.raise_for_status()
        return response.json().get("coding_agent", {})

    async def set_coding_agent(self, provider: str) -> dict[str, Any]:
        response = await self.client.post(
            "/api/coding-agents/default",
            json={"provider": provider},
        )
        response.raise_for_status()
        return response.json().get("coding_agent", {})

    async def set_coding_workspace(self, workspace_path: str) -> dict[str, Any]:
        response = await self.client.post(
            "/api/coding-agents/workspace",
            json={"workspace_path": workspace_path},
        )
        response.raise_for_status()
        return response.json().get("coding_agent", {})

    async def set_coding_network(self, enabled: bool) -> dict[str, Any]:
        response = await self.client.post(
            "/api/coding-agents/network",
            json={"enabled": bool(enabled)},
        )
        response.raise_for_status()
        return response.json().get("coding_agent", {})

    async def list_coding_tasks(self, status: str = "active") -> list[dict[str, Any]]:
        response = await self.client.get("/api/coding-tasks", params={"status": status})
        response.raise_for_status()
        return response.json().get("tasks", [])

    async def clear_coding_task_history(self) -> dict[str, Any]:
        response = await self.client.delete("/api/coding-tasks")
        response.raise_for_status()
        return response.json()

    async def start_coding_task(self, prompt: str) -> dict[str, Any]:
        response = await self.client.post("/api/coding-tasks", json={"prompt": prompt})
        response.raise_for_status()
        return response.json()

    async def get_coding_task(self, task_id: str) -> dict[str, Any]:
        response = await self.client.get(f"/api/coding-tasks/{task_id}")
        response.raise_for_status()
        return response.json().get("task", {})

    async def delete_coding_task(self, task_id: str) -> dict[str, Any]:
        response = await self.client.delete(f"/api/coding-tasks/{task_id}")
        response.raise_for_status()
        return response.json()

    async def get_coding_logs(self, task_id: str, limit: int = 50) -> list[dict[str, Any]]:
        response = await self.client.get(
            f"/api/coding-tasks/{task_id}/logs",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json().get("logs", [])

    async def reply_to_coding_task(self, task_id: str, message: str) -> dict[str, Any]:
        response = await self.client.post(
            f"/api/coding-tasks/{task_id}/reply",
            json={"message": message},
        )
        response.raise_for_status()
        return response.json()

    async def reply_to_latest_coding_task(self, message: str) -> dict[str, Any]:
        response = await self.client.post(
            "/api/coding-tasks/latest/reply",
            json={"message": message},
        )
        response.raise_for_status()
        return response.json()

    async def cancel_coding_task(self, task_id: str) -> dict[str, Any]:
        response = await self.client.post(f"/api/coding-tasks/{task_id}/cancel")
        response.raise_for_status()
        return response.json()

    async def set_coding_task_network(self, task_id: str, enabled: bool) -> dict[str, Any]:
        response = await self.client.post(
            f"/api/coding-tasks/{task_id}/network",
            json={"enabled": bool(enabled)},
        )
        response.raise_for_status()
        return response.json()

    async def subscribe_coding_tasks_to_channel(self, channel_type: str) -> dict[str, Any]:
        response = await self.client.post(
            "/api/coding-tasks/subscribe-channel",
            json={"channel_type": channel_type},
        )
        response.raise_for_status()
        return response.json()


class CloudflareTunnel:
    def __init__(self, local_url: str, events: EventLog, command: str = "cloudflared"):
        self.local_url = local_url.rstrip("/")
        self.events = events
        self.command = command
        self.process: asyncio.subprocess.Process | None = None
        self.reader_tasks: list[asyncio.Task] = []
        self.url_future: asyncio.Future[str] | None = None

    async def start(self, timeout: float = 45) -> str:
        executable = shutil.which(self.command)
        if not executable:
            raise RuntimeError(
                "cloudflared is not installed or not on PATH. Install it with: brew install cloudflare/cloudflare/cloudflared"
            )

        self.url_future = asyncio.get_running_loop().create_future()
        self.process = await asyncio.create_subprocess_exec(
            executable,
            "tunnel",
            "--url",
            self.local_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.reader_tasks = [
            asyncio.create_task(self._read_stream(self.process.stdout)),
            asyncio.create_task(self._read_stream(self.process.stderr)),
        ]

        try:
            tunnel_url = await asyncio.wait_for(self.url_future, timeout=timeout)
        except Exception:
            await self.close()
            raise

        await self.events.emit("Tunnel ready", tunnel_url, "magenta")
        return tunnel_url

    async def _read_stream(self, stream: asyncio.StreamReader | None) -> None:
        if stream is None or self.url_future is None:
            return

        while True:
            raw_line = await stream.readline()
            if not raw_line:
                break

            line = raw_line.decode(errors="ignore").strip()
            match = TRYCLOUDFLARE_URL_RE.search(line)
            if match and not self.url_future.done():
                self.url_future.set_result(match.group(0))

    async def close(self) -> None:
        for task in self.reader_tasks:
            task.cancel()
        if self.reader_tasks:
            await asyncio.gather(*self.reader_tasks, return_exceptions=True)

        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()


class ChannelManager:
    def __init__(
        self,
        api_base: str,
        events: EventLog,
        ui: Any,
        cloudflared_command: str,
        tunnel_enabled: bool = True,
    ):
        self.api_base = api_base
        self.events = events
        self.ui = ui
        self.cloudflared_command = cloudflared_command
        self.tunnel_enabled = tunnel_enabled
        self.tunnel: CloudflareTunnel | None = None
        self.tunnel_url: str | None = None
        self.connected_channels: set[str] = set()

    async def connect(self, channel_type: str) -> str:
        channel = normalize_channel_type(channel_type)
        if not channel:
            raise RuntimeError("Usage: /connect <channel_type>\nSupported now: telegram")

        if channel == "telegram":
            return await self.connect_telegram()

        raise RuntimeError(
            f"Channel '{channel}' is not supported yet.\n"
            "Supported now: telegram"
        )

    async def connect_telegram(self) -> str:
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        webhook_secret = os.getenv("TELEGRAM_WEBHOOK_SECRET")
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in the environment.")
        if not self.tunnel_enabled:
            raise RuntimeError("Tunnels are disabled for this CLI run.")

        if self.ui is not None:
            self.ui.set_status("opening public tunnel...", "busy")

        tunnel_url = await self.ensure_tunnel()
        webhook_url = f"{tunnel_url}{TELEGRAM_WEBHOOK_PATH}"

        if self.ui is not None:
            self.ui.set_status("registering telegram channel...", "busy")
        await set_telegram_webhook(token, webhook_url, webhook_secret, self.events)

        self.connected_channels.add("telegram")
        if self.ui is not None:
            self.ui.set_channel_connected("telegram")
            self.ui.set_status("telegram connected", "ready")

        await self.subscribe_coding_updates("telegram")

        return "Telegram channel connected."

    async def subscribe_coding_updates(self, channel_type: str) -> None:
        try:
            async with httpx.AsyncClient(base_url=self.api_base, timeout=15) as client:
                response = await client.post(
                    "/api/coding-tasks/subscribe-channel",
                    json={"channel_type": channel_type},
                )
                response.raise_for_status()
                result = response.json()
        except Exception as exc:
            await self.events.emit("Coding notifications", f"Could not subscribe {channel_type}: {exc}", "yellow")
            return

        message = str(result.get("message") or "").strip()
        task_count = int(result.get("task_count") or 0)
        style = "green" if task_count else "yellow"
        await self.events.emit("Coding notifications", message, style)

    async def ensure_tunnel(self) -> str:
        if self.tunnel_url:
            return self.tunnel_url

        self.tunnel = CloudflareTunnel(
            self.api_base,
            self.events,
            command=self.cloudflared_command,
        )
        self.tunnel_url = await self.tunnel.start()
        if self.ui is not None:
            self.ui.set_status("waiting for public tunnel...", "busy")
        await wait_for_public_tunnel(self.tunnel_url, self.events)
        return self.tunnel_url

    async def close(self) -> None:
        if self.tunnel is not None:
            await self.tunnel.close()


async def set_telegram_webhook(
    token: str,
    webhook_url: str,
    secret: str | None,
    events: EventLog,
    attempts: int = 8,
) -> dict[str, Any]:
    payload = {
        "url": webhook_url,
        "allowed_updates": [
            "message",
            "edited_message",
            "channel_post",
            "edited_channel_post",
        ],
        "drop_pending_updates": False,
    }
    if secret:
        if not TELEGRAM_SECRET_RE.fullmatch(secret):
            raise RuntimeError(
                "TELEGRAM_WEBHOOK_SECRET is invalid. Telegram allows only A-Z, a-z, 0-9, underscore, and hyphen, with length 1-256."
            )
        payload["secret_token"] = secret

    last_error = None
    async with httpx.AsyncClient(timeout=30) as client:
        for attempt in range(1, attempts + 1):
            try:
                response = await client.post(
                    f"{TELEGRAM_API_BASE}/bot{token}/setWebhook",
                    json=payload,
                )
                if response.status_code >= 400:
                    last_error = (
                        f"HTTP {response.status_code}: {response.text}"
                    )
                else:
                    data = response.json()
                    if data.get("ok"):
                        await events.emit("Telegram ready", "Channel registration completed.", "green")
                        return data
                    last_error = str(data)
            except Exception as exc:
                last_error = str(exc)

            if attempt < attempts:
                delay = min(2 * attempt, 10)
                await events.emit(
                    "Channel registration retry",
                    f"Attempt {attempt}/{attempts} failed: {last_error}\nRetrying in {delay}s.",
                    "yellow",
                )
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"Telegram setWebhook failed after {attempts} attempts: {last_error}"
    )


async def wait_for_public_tunnel(
    tunnel_url: str,
    events: EventLog,
    timeout: float = 10,
) -> None:
    deadline = time.monotonic() + timeout

    async with httpx.AsyncClient(timeout=5, follow_redirects=True) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(tunnel_url)
                if response.status_code < 500:
                    await events.emit("Tunnel reachable", tunnel_url, "magenta")
                    return
            except Exception:
                pass

            await asyncio.sleep(1)

    await events.emit(
        "Tunnel reachability pending",
        (
            f"Tunnel is still propagating after {timeout:.0f}s.\n"
            "Continuing to channel registration now."
        ),
        "yellow",
    )


def format_event_details(event: dict[str, Any]) -> str:
    data = event.get("data") or {}
    parts = []
    if data.get("job_id"):
        parts.append(f"job={str(data['job_id'])[:8]}")
    if data.get("task_id"):
        parts.append(f"task={str(data['task_id'])[:12]}")
    if data.get("provider"):
        parts.append(f"provider={data['provider']}")
    if data.get("status"):
        parts.append(f"status={data['status']}")
    if data.get("channel"):
        parts.append(f"channel={data['channel']}")
    if data.get("stage"):
        parts.append(f"stage={data['stage']}")
    if data.get("tool_name"):
        parts.append(f"tool={data['tool_name']}")
    if data.get("skills"):
        skills = ",".join(str(skill) for skill in data["skills"])
        parts.append(f"skills={skills}")
    if data.get("username"):
        parts.append(f"from={data['username']}")
    if data.get("elapsed") is not None:
        parts.append(f"{data['elapsed']}s")

    metrics = data.get("metrics") or {}
    if metrics:
        retrieval = token_pair(metrics, "retrieval")
        generation = token_pair(metrics, "generation")
        if retrieval:
            parts.append(f"retrieval={retrieval}")
        if generation:
            parts.append(f"generation={generation}")

    contexts = data.get("contexts") or {}
    if contexts:
        context_bits = ", ".join(f"{key}:{value}" for key, value in contexts.items())
        parts.append(f"contexts={context_bits}")

    return "  ".join(parts)


def event_prefix(kind: str) -> str:
    if kind == "response":
        return "agent "
    if kind == "error":
        return "error "
    if kind == "request":
        return "user "
    if kind == "job":
        return "job   "
    if kind == "coding":
        return "code  "
    return "event "


def progress_replace_key(event: dict[str, Any]) -> str | None:
    kind = str(event.get("kind") or "").lower()
    data = event.get("data") or {}
    if data.get("replace_key"):
        return str(data["replace_key"])
    job_id = data.get("job_id")
    if kind == "job" and job_id:
        return f"job:{job_id}:progress"
    return None


def normalize_event_kind(kind: str) -> str:
    cleaned = str(kind or "event").strip().lower()
    return {
        "response": "agent",
        "request": "user",
        "warning": "notice",
        "system": "event",
        "telegram": "event",
        "job": "notice",
        "coding": "coding",
    }.get(cleaned, cleaned if cleaned in {"agent", "coding", "error", "event", "notice", "ready", "tunnel", "user"} else "event")


def normalize_status_kind(kind: str) -> str:
    cleaned = str(kind or "notice").strip().lower()
    if cleaned in {"ready", "busy", "error", "notice"}:
        return cleaned
    if cleaned in {"agent", "tunnel", "user"}:
        return "busy"
    return "notice"


def event_kind_from_style(style: str) -> str:
    return {
        "green": "ready",
        "yellow": "notice",
        "red": "error",
        "magenta": "tunnel",
        "bright_magenta": "coding",
    }.get(style, "event")


def status_style(kind: str) -> str:
    return {
        "ready": "bold #22c55e",
        "busy": "bold #fbbf24",
        "error": "bold #f87171",
        "notice": "bold #60a5fa",
    }.get(normalize_status_kind(kind), "bold #60a5fa")


def event_style(kind: str) -> str:
    return {
        "agent": "bold #2dd4bf",
        "coding": "bold #a78bfa",
        "error": "bold #f87171",
        "event": "bold #94a3b8",
        "notice": "bold #fbbf24",
        "ready": "bold #22c55e",
        "tunnel": "bold #c084fc",
        "user": "bold #60a5fa",
    }.get(normalize_event_kind(kind), "bold #94a3b8")


def model_label() -> str:
    model = (
        os.getenv("QUARQ_MODEL_LABEL")
        or os.getenv("GENERATION_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1"
    ).strip()
    effort = (
        os.getenv("QUARQ_REASONING_EFFORT")
        or os.getenv("REASONING_EFFORT")
        or ""
    ).strip()
    return f"{model} {effort}".strip()


def load_cli_agent_name() -> str:
    try:
        return str(load_agent_config().get("agent_name") or "Argus").strip() or "Argus"
    except Exception:
        return os.getenv("AGENT_NAME", "Argus").strip() or "Argus"


def normalize_channel_type(channel_type: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", str(channel_type or "").strip().lower())


def normalize_channel_list(channels: Any) -> list[str]:
    if isinstance(channels, str):
        channels = re.split(r"[,\s]+", channels.strip())
    if not isinstance(channels, list):
        channels = []

    normalized = []
    seen = set()
    for channel in channels:
        value = normalize_channel_type(channel)
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    return normalized


def channel_summary(channels: set[str] | list[str]) -> str:
    normalized = normalize_channel_list(list(channels))
    return ", ".join(normalized) if normalized else "none"


def load_cli_config() -> dict[str, Any]:
    defaults = {"startup_channels": []}
    if not CLI_CONFIG_PATH.exists():
        return defaults

    try:
        data = json.loads(CLI_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    if not isinstance(data, dict):
        return defaults

    return {
        "startup_channels": normalize_channel_list(data.get("startup_channels") or []),
    }


def save_cli_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "startup_channels": normalize_channel_list(config.get("startup_channels") or []),
    }
    CLI_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = CLI_CONFIG_PATH.with_suffix(f"{CLI_CONFIG_PATH.suffix}.tmp")
    temp_path.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(CLI_CONFIG_PATH)
    return normalized


def format_set_default_response(config: dict[str, Any]) -> str:
    startup_channels = normalize_channel_list(config.get("startup_channels") or [])
    if startup_channels:
        return f"Startup channels: {', '.join(startup_channels)}"
    return "Startup channels cleared."


def compact_path(path: Path) -> str:
    try:
        path = path.resolve()
    except OSError:
        pass

    home = Path.home()
    try:
        return f"~/{path.relative_to(home)}"
    except ValueError:
        return str(path)


def welcome_block() -> dict[str, Any]:
    return {
        "key": "welcome",
        "timestamp": "",
        "kind": "event",
        "title": "Getting Started",
        "details": "",
        "body": local_help_text(intro=True),
    }


COMMAND_OPTIONS = [
    {
        "name": "/help",
        "insert": "/help",
        "description": "show available commands",
    },
    {
        "name": "/status",
        "insert": "/status",
        "description": "show current API and channel configuration",
    },
    {
        "name": "/tools",
        "insert": "/tools",
        "description": "list enabled native and cloud tools",
    },
    {
        "name": "/which-tool",
        "insert": "/which-tool ",
        "description": "show which tool fits a task",
    },
    {
        "name": "/cloud-tools",
        "insert": "/cloud-tools",
        "description": "list cloud tools available to enable",
    },
    {
        "name": "/add-tool",
        "insert": "/add-tool ",
        "description": "enable a cloud tool",
    },
    {
        "name": "/remove-tool",
        "insert": "/remove-tool ",
        "description": "disable a cloud tool",
    },
    {
        "name": "/coding",
        "insert": "/coding ",
        "description": "start a new coding task or list recent tasks",
    },
    {
        "name": "/coding-new",
        "insert": "/coding-new ",
        "description": "start a fresh coding session",
    },
    {
        "name": "/coding-continue",
        "insert": "/coding-continue ",
        "description": "continue the latest coding session",
    },
    {
        "name": "/coding-tasks",
        "insert": "/coding-tasks",
        "description": "list recent coding task ids",
    },
    {
        "name": "/coding-agents",
        "insert": "/coding-agents",
        "description": "list coding agents and current default",
    },
    {
        "name": "/coding-use",
        "insert": "/coding-use codex",
        "description": "select default coding agent",
    },
    {
        "name": "/coding-workspace",
        "insert": "/coding-workspace ",
        "description": "set default coding workspace",
    },
    {
        "name": "/coding-network",
        "insert": "/coding-network ",
        "description": "show or set coding network access",
    },
    {
        "name": "/coding-allow-network",
        "insert": "/coding-allow-network ",
        "description": "allow network access for a coding task",
    },
    {
        "name": "/coding-status",
        "insert": "/coding-status ",
        "description": "show coding task status",
    },
    {
        "name": "/coding-log",
        "insert": "/coding-log ",
        "description": "show recent coding task logs",
    },
    {
        "name": "/coding-reply",
        "insert": "/coding-reply ",
        "description": "reply to a coding task",
    },
    {
        "name": "/coding-cancel",
        "insert": "/coding-cancel ",
        "description": "cancel a coding task",
    },
    {
        "name": "/coding-delete",
        "insert": "/coding-delete ",
        "description": "delete a completed coding task",
    },
    {
        "name": "/coding-clear",
        "insert": "/coding-clear",
        "description": "delete completed coding task history",
    },
    {
        "name": "/connect",
        "insert": "/connect telegram ",
        "description": "connect a channel on demand",
    },
    {
        "name": "/wipe",
        "insert": "/wipe",
        "description": "clear local memories",
    },
    {
        "name": "/quit",
        "insert": "/quit",
        "description": "stop the local control console",
    },
    {
        "name": "set-default start-channel",
        "insert": "set-default start-channel telegram ",
        "description": "auto-connect channels on future startup",
    },
]

TASK_ID_COMMANDS = {
    "/coding-status": {"needs_message": False, "description": "show this task status"},
    "/coding-log": {"needs_message": False, "description": "show this task log"},
    "/coding-reply": {"needs_message": True, "description": "reply to this task"},
    "/coding-cancel": {"needs_message": False, "description": "cancel this task"},
    "/coding-delete": {"needs_message": False, "description": "delete this task"},
    "/coding-rm": {"needs_message": False, "description": "delete this task"},
    "/coding-continue": {"needs_message": True, "description": "continue this task"},
    "/coding-network": {"needs_message": True, "description": "set this task network access"},
    "/coding-allow-network": {"needs_message": False, "description": "allow network for this task"},
}


def recent_coding_task_options(command: str, partial: str = "") -> list[dict[str, str]]:
    partial = str(partial or "").strip().lower()
    try:
        tasks = CodingTaskStore(load_settings().task_root).list_tasks(limit=20)
    except Exception:
        return []

    options = []
    for task in tasks:
        task_id = str(task.get("id") or "")
        if not task_id:
            continue
        prompt = str(task.get("prompt") or "").splitlines()[0].strip()
        haystack = " ".join(
            [
                task_id,
                str(task.get("status") or ""),
                prompt,
            ]
        ).lower()
        if partial and partial not in haystack:
            continue

        needs_message = bool(TASK_ID_COMMANDS.get(command, {}).get("needs_message"))
        insert = f"{command} {task_id}{' ' if needs_message else ''}"
        short_prompt = prompt[:64] + ("..." if len(prompt) > 64 else "")
        options.append(
            {
                "name": task_id,
                "insert": insert,
                "description": (
                    f"[{task.get('status')}] {TASK_ID_COMMANDS[command]['description']}"
                    + (f" - {short_prompt}" if short_prompt else "")
                ),
                "kind": "task",
            }
        )

    return options[:8]


def command_suggestions(text: str) -> list[dict[str, str]]:
    lines = str(text or "").splitlines()
    raw_first_line = lines[0] if lines else ""
    first_line = raw_first_line.strip()
    if not first_line:
        return []

    if first_line.startswith("/"):
        command, separator, rest = raw_first_line.partition(" ")
        command = command.strip().lower()
        if command in TASK_ID_COMMANDS and separator:
            rest_parts = rest.strip().split(maxsplit=1)
            if len(rest_parts) > 1:
                return []
            first_arg = rest_parts[0] if rest_parts else ""
            if command == "/coding-network" and not first_arg.startswith("code_"):
                network_options = [
                    {
                        "name": "on",
                        "insert": "/coding-network on",
                        "description": "allow network for new coding tasks",
                        "kind": "command",
                    },
                    {
                        "name": "off",
                        "insert": "/coding-network off",
                        "description": "block network for new coding tasks",
                        "kind": "command",
                    },
                ]
                if first_arg:
                    return [
                        option
                        for option in network_options
                        if option["name"].startswith(first_arg.lower())
                    ]
                return network_options + recent_coding_task_options(command, first_arg)[:6]
            if not first_arg or first_arg.startswith("code_"):
                return recent_coding_task_options(command, first_arg)
            return []

        query = first_line.lower()
        return [
            option
            for option in COMMAND_OPTIONS
            if option["name"].startswith("/") and option["name"].startswith(query)
        ] or [
            option
            for option in COMMAND_OPTIONS
            if option["name"].startswith("/")
        ]

    if "set-default".startswith(first_line.lower()) or first_line.lower().startswith("set-default"):
        return [
            option
            for option in COMMAND_OPTIONS
            if option["name"].startswith("set-default")
        ]

    return []


def render_command_suggestions(suggestions: list[dict[str, str]], selected_index: int = 0) -> Text:
    text = Text()
    for index, option in enumerate(suggestions[:8]):
        if index:
            text.append("\n")
        selected = index == selected_index
        name_style = "bold #111318 on #60a5fa" if selected else "bold #22d3ee"
        desc_style = "#111318 on #60a5fa" if selected else "#8b949e"
        prefix = "› " if selected else "  "
        text.append(prefix, style=name_style if selected else "#6b7280")
        text.append(option["name"], style=name_style)
        text.append("  ")
        text.append(option["description"], style=desc_style)
    text.append("\nCtrl+N/P to choose · Tab to insert", style="#6b7280")
    return text


def local_help_text(intro: bool = False) -> str:
    rows = []
    if intro:
        rows.extend(["Describe a task, ask a question, or try one of these commands:", ""])
    rows.extend(
        [
            "- `/help` - show available commands",
            "- `/status` - show current API and channel configuration",
            "- `/tools` - list enabled native and cloud tools",
            "- `/which-tool <task>` - show which tool fits a task",
            "- `/cloud-tools` - list cloud tools available to enable",
            "- `/add-tool <tool>` - enable a cloud tool",
            "- `/remove-tool <tool>` - disable a cloud tool",
            "- `/coding <task>` - start a new coding task",
            "- `/coding-new <task>` - start a fresh coding session",
            "- `/coding-continue <message>` - continue the latest coding session",
            "- `/coding` or `/coding-tasks` - list recent coding task ids",
            "- `/coding-agents` - list coding agents and current default",
            "- `/coding-use <provider>` - select the default coding agent",
            "- `/coding-workspace <path>` - set the default coding workspace",
            "- `/coding-network [on|off]` - show or set default coding network access",
            "- `/coding-network <id> on|off` - set network access for a coding task",
            "- `/coding-allow-network <id>` - allow network access for a coding task",
            "- `/coding-status <id>` - show coding task status",
            "- `/coding-log <id>` - show recent coding task logs",
            "- `/coding-reply <id> <message>` - reply to a coding task",
            "- `/coding-cancel <id>` - cancel a coding task",
            "- `/coding-delete <id>` - delete a completed/failed/cancelled coding task",
            "- `/coding-clear` - delete all completed/failed/cancelled coding task history",
            "- `/connect telegram` - start a channel connection on demand",
            "- `set-default start-channel telegram` - auto-connect a channel on future startup",
            "- `set-default start-channel none` - clear startup channel connections",
            "- `/wipe` - clear local memories",
            "- `/quit` - stop the local control console",
            "",
            f"Enabled cloud tools: {format_slug_list(load_enabled_cloud_tools())}",
            "Expand tools with `/cloud-tools`, then `/add-tool <tool>`.",
        ]
    )
    return "\n".join(rows)


def parse_on_off(value: str) -> bool | None:
    lowered = str(value or "").strip().lower()
    if lowered in {"on", "true", "1", "yes", "enable", "enabled", "allow", "allowed"}:
        return True
    if lowered in {"off", "false", "0", "no", "disable", "disabled", "block", "blocked"}:
        return False
    return None


def format_coding_task(task: dict[str, Any]) -> str:
    if not task:
        return "Coding task not found."
    rows = [
        f"Task id: {task.get('id')}",
        f"Status: {task.get('status')}",
        f"Provider: {task.get('provider')}",
        f"Workspace: {task.get('workspace_path')}",
        f"Network: {'on' if task.get('network_access', True) else 'off'}",
    ]
    if task.get("approval_request"):
        rows.append("Approval needed: yes")
    if task.get("provider_thread_id"):
        rows.append("Codex session: connected")
    if task.get("restart_provider_session"):
        rows.append("Codex session: will restart on next continuation")
    if task.get("current_activity"):
        rows.append(f"Current activity: {task.get('current_activity')}")
    if task.get("changed_files"):
        rows.append("Changed files:\n" + "\n".join(f"- {path}" for path in task["changed_files"]))
    if task.get("error"):
        rows.append(f"Error: {task.get('error')}")
    if task.get("result_summary"):
        rows.append(f"Summary: {task.get('result_summary')}")
    return "\n".join(rows)


def format_coding_agent_config(config: dict[str, Any]) -> str:
    providers = config.get("providers") or []
    rows = [
        f"Default coding agent: {config.get('default_provider_label')} (`{config.get('default_provider')}`)",
        f"Workspace: {config.get('workspace_root')}",
        f"Approval policy: {config.get('approval_policy')}",
        f"Network access: {'on' if config.get('network_access') else 'off'}",
        f"Active tasks: {config.get('active_tasks', 0)}",
        "",
        "Available coding agents:",
    ]
    rows.extend(
        f"- {item['label']} (`{item['slug']}`) [{item['status']}] - {item['description']}"
        for item in providers
    )
    rows.append("")
    rows.append("Use `/coding-use codex`, `/coding-workspace <path>`, and `/coding-network on|off` to change defaults.")
    return "\n".join(rows)


def format_coding_tasks(tasks: list[dict[str, Any]], active_only: bool = True) -> str:
    if not tasks:
        return (
            "No active coding tasks. Start one with `/coding <task>`."
            if active_only
            else "No coding tasks yet. Start one with `/coding <task>`."
        )
    rows = ["Active coding tasks:" if active_only else "Recent coding tasks:"]
    for task in tasks[:20]:
        prompt = str(task.get("prompt") or "").splitlines()[0][:90]
        marker = " needs input" if task.get("approval_request") or task.get("status") == "waiting_user" else ""
        network = "on" if task.get("network_access", True) else "off"
        rows.append(f"- {task.get('id')} [{task.get('status')}] net={network}{marker} {prompt}")
    rows.append("")
    rows.append("Continue latest with `/coding-continue <message>`. Start fresh with `/coding-new <task>`.")
    rows.append("Inspect with `/coding-status <id>` or `/coding-log <id>`. Reply to a specific task with `/coding-reply <id> <message>`.")
    rows.append("Delete history with `/coding-delete <id>` or `/coding-clear`.")
    rows.append("For id commands, type the command plus a space, choose a task with Ctrl+N/P, then press Tab.")
    return "\n".join(rows)


def format_coding_logs(logs: list[dict[str, Any]]) -> str:
    if not logs:
        return "No coding task logs found."
    rows = []
    for event in logs:
        rows.append(
            f"- {event.get('time', '')} {event.get('title', 'Log')}: "
            f"{str(event.get('message') or '').strip()}"
        )
    return "\n".join(rows)


def render_textual_block(block: dict[str, Any]) -> Group:
    kind = normalize_event_kind(str(block.get("kind") or "event"))
    header = Text(overflow="fold", no_wrap=False)
    timestamp = str(block.get("timestamp") or "")
    if timestamp:
        header.append(timestamp, style="#6b7280")
        header.append("  ")
    header.append(f"{kind:<6}", style=event_style(kind))
    header.append("  ")
    header.append(str(block.get("title") or ""), style="bold #e5e7eb")

    details = str(block.get("details") or "").strip()
    if details:
        header.append("  ")
        append_textual_details(header, details)

    renderables: list[Any] = [header]
    body = str(block.get("body") or "").strip()
    if body:
        renderables.append(Padding(Markdown(body), (0, 0, 0, 2)))
    renderables.append(Rule(style="#3b414a"))
    return Group(*renderables)


def transcript_render_width(log: Any) -> int:
    for attr in ("scrollable_content_region", "content_region", "region"):
        region = getattr(log, attr, None)
        width = getattr(region, "width", 0)
        if width:
            return max(1, int(width) - 2)
    return max(1, shutil.get_terminal_size((100, 32)).columns - 4)


def append_textual_details(text: Text, details: str) -> None:
    parts = [part.strip() for part in str(details or "").split("  ") if part.strip()]
    for index, part in enumerate(parts):
        if index:
            text.append("  ", style="#6b7280")
        if "=" not in part:
            text.append(part, style="#38bdf8")
            continue

        key, value = part.split("=", 1)
        value_style = "#a78bfa" if key in {"retrieval", "generation"} else "#38bdf8"
        text.append(key, style="#94a3b8")
        text.append("=", style="#6b7280")
        text.append(value, style=value_style)


def call_scroll_method(widget: Any, method_name: str) -> None:
    method = getattr(widget, method_name, None)
    if method is None:
        return
    try:
        method(animate=False)
    except TypeError:
        method()


def is_scroll_at_end(widget: Any) -> bool:
    scroll_y = getattr(widget, "scroll_y", None)
    max_scroll_y = getattr(widget, "max_scroll_y", None)
    if scroll_y is None or max_scroll_y is None:
        return False
    return scroll_y >= max_scroll_y


def render_details(details: str) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    parts = [part.strip() for part in str(details or "").split("  ") if part.strip()]
    for index, part in enumerate(parts):
        if index:
            fragments.append(("class:md.dim", "  "))
        if "=" not in part:
            fragments.append(("class:event.detail.value", part))
            continue

        key, value = part.split("=", 1)
        value_style = "class:event.detail.metric" if key in {"retrieval", "generation"} else "class:event.detail.value"
        fragments.extend(
            [
                ("class:event.detail.key", key),
                ("class:md.dim", "="),
                (value_style, value),
            ]
        )
    return fragments


def render_markdown_body(body: str) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    in_code_block = False

    for raw_line in str(body or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            language = stripped.strip("`").strip()
            if in_code_block and language:
                fragments.extend([("class:md.dim", "  code "), ("class:md.code", language), ("", "\n")])
            continue

        if in_code_block:
            fragments.extend([("class:md.dim", "  │ "), ("class:md.codeblock", line), ("", "\n")])
            continue

        if not stripped:
            fragments.append(("", "\n"))
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            fragments.extend([("class:md.dim", "  "), ("class:md.heading", heading.group(2)), ("", "\n")])
            continue

        numbered = re.match(r"^(\s*)(\d+)\.\s+(.+)$", line)
        if numbered:
            indent, number, text = numbered.groups()
            fragments.append(("class:md.dim", "  " + indent))
            fragments.append(("class:md.list.marker", f"{number}. "))
            fragments.extend(render_inline_markdown(text))
            fragments.append(("", "\n"))
            continue

        bullet = re.match(r"^(\s*)[-*]\s+(.+)$", line)
        if bullet:
            indent, text = bullet.groups()
            fragments.append(("class:md.dim", "  " + indent))
            fragments.append(("class:md.list.marker", "• "))
            fragments.extend(render_inline_markdown(text))
            fragments.append(("", "\n"))
            continue

        key_value = re.match(r"^([A-Za-z_][A-Za-z0-9_ -]{0,40}):\s+(.+)$", stripped)
        if key_value and not stripped.startswith(("http://", "https://")):
            key, value = key_value.groups()
            fragments.extend([("class:md.dim", "  "), ("class:event.detail.key", key), ("class:md.dim", ": ")])
            fragments.extend(render_inline_markdown(value))
            fragments.append(("", "\n"))
            continue

        fragments.append(("class:md.dim", "  "))
        fragments.extend(render_inline_markdown(line))
        fragments.append(("", "\n"))

    return fragments


def render_inline_markdown(text: str) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = []
    token_re = re.compile(r"(\*\*.+?\*\*|`[^`]+`|https?://[^\s)]+)")
    cursor = 0
    for match in token_re.finditer(text):
        if match.start() > cursor:
            fragments.append(("class:md", text[cursor:match.start()]))

        token = match.group(0)
        if token.startswith("**") and token.endswith("**"):
            fragments.append(("class:md.bold", token[2:-2]))
        elif token.startswith("`") and token.endswith("`"):
            fragments.append(("class:md.code", token[1:-1]))
        else:
            fragments.append(("class:md.url", token))
        cursor = match.end()

    if cursor < len(text):
        fragments.append(("class:md", text[cursor:]))
    return fragments


def fragment_line_count(fragments: list[tuple[str, str]]) -> int:
    text = "".join(fragment for _, fragment in fragments)
    return max(1, text.count("\n") + 1)


def split_fragments_into_lines(
    fragments: list[tuple[str, str]]
) -> list[list[tuple[str, str]]]:
    lines: list[list[tuple[str, str]]] = [[]]
    for style, text in fragments:
        parts = str(text).split("\n")
        for index, part in enumerate(parts):
            if part:
                lines[-1].append((style, part))
            if index < len(parts) - 1:
                lines.append([])
    return lines or [[]]


def wrap_fragment_lines(
    lines: list[list[tuple[str, str]]],
    width: int,
) -> list[list[tuple[str, str]]]:
    wrapped: list[list[tuple[str, str]]] = []
    width = max(1, width)

    for line in lines:
        if not line:
            wrapped.append([])
            continue

        current: list[tuple[str, str]] = []
        current_width = 0
        for style, text in line:
            remaining = str(text)
            while remaining:
                available = width - current_width
                if available <= 0:
                    wrapped.append(current)
                    current = []
                    current_width = 0
                    available = width

                chunk = remaining[:available]
                current.append((style, chunk))
                current_width += len(chunk)
                remaining = remaining[available:]

                if remaining and current_width >= width:
                    wrapped.append(current)
                    current = []
                    current_width = 0

        wrapped.append(current)

    return wrapped or [[]]


def separator_text() -> str:
    width = shutil.get_terminal_size((100, 24)).columns
    return "─" * max(32, width - 3)


def token_pair(metrics: dict[str, Any], prefix: str) -> str:
    in_tokens = metrics.get(f"{prefix}_in")
    out_tokens = metrics.get(f"{prefix}_out")
    if in_tokens is None and out_tokens is None:
        return ""
    return f"{in_tokens or 0}->{out_tokens or 0}"


def print_header(console: Console, api_base: str) -> None:
    console.clear()
    console.print(Text(load_cli_agent_name(), style="bold white"))
    console.print(
        Text(
            f"api {api_base}  commands /help /status /tools /cloud-tools /coding /coding-network /connect /wipe /quit",
            style="dim",
        )
    )
    console.print(Text("-" * min(console.width, 120), style="bright_black"))
    console.print(Text("Write in the bottom input row. Channel events appear above it.", style="dim"))
    console.print()


def build_terminal_ui(api_base: str) -> TextualTerminalUi | None:
    if QuarqTextualApp is None:
        return None
    return TextualTerminalUi(api_base)


def start_api_server(host: str, port: int, log_level: str) -> subprocess.Popen:
    os.environ.setdefault("USER_ID", "local_cli_user")
    os.environ.setdefault("QUARQ_LAUNCH_CWD", str(LAUNCH_CWD))

    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
        "--no-access-log",
    ]
    return subprocess.Popen(
        command,
        cwd=BASE_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


async def stop_api_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return

    server.terminate()
    try:
        await asyncio.to_thread(server.wait, timeout=5)
    except subprocess.TimeoutExpired:
        server.kill()
        await asyncio.to_thread(server.wait)


async def wait_for_api(api: QuarqApiClient, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            await api.health()
            return
        except Exception as exc:
            last_error = exc
            await asyncio.sleep(0.25)

    raise RuntimeError(f"API did not become ready within {timeout:.1f}s: {last_error}")


async def poll_job_until_complete(
    api: QuarqApiClient,
    job_id: str,
    ui: TerminalUi | None,
) -> dict[str, Any]:
    last_status_key = None

    while True:
        job = await api.get_job(job_id)
        status = str(job.get("status") or "unknown")
        stage = str(job.get("stage") or status)
        message = str(job.get("message") or "").strip()
        tool_name = str(job.get("tool_name") or "").strip()
        status_key = (status, stage, message, tool_name)

        if ui is not None and status_key != last_status_key:
            if status == "completed":
                ui.set_status("ready", "ready")
            elif status == "failed":
                ui.set_status("job failed", "error")
            elif tool_name:
                ui.set_status(f"using tool: {tool_name}", "busy")
            elif message:
                ui.set_status(message.lower(), "busy")
            else:
                ui.set_status(stage.replace("_", " "), "busy")
            last_status_key = status_key

        if status in {"completed", "failed"}:
            return job

        await asyncio.sleep(JOB_POLL_INTERVAL)


async def cli_input_loop(
    api: QuarqApiClient,
    events: EventLog,
    console: Console,
    ui: TerminalUi | None,
    channel_manager: ChannelManager,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        try:
            if ui is None:
                user_input = await asyncio.to_thread(console.input, "\n[bold white]›[/bold white] ")
            else:
                user_input = await ui.read_input()
        except (EOFError, KeyboardInterrupt):
            stop_event.set()
            return

        prompt = user_input.strip()
        if not prompt:
            continue

        words = prompt.split()
        command = words[0].lower() if words else ""

        if prompt in {"/quit", "/exit"}:
            stop_event.set()
            return

        if command == "/help":
            await events.emit("Available commands", local_help_text(), "green")
            continue

        if command == "/status":
            status = await api.health()
            details = [
                format_status(status),
                f"connected_channels: {channel_summary(channel_manager.connected_channels)}",
                f"startup_channels: {channel_summary(load_cli_config().get('startup_channels', []))}",
            ]
            await events.emit("Control status", "\n".join(part for part in details if part), "green")
            continue

        if command in {"/coding", "/coding-new", "/coding-tasks"}:
            task_prompt = prompt.partition(" ")[2].strip()
            try:
                if not task_prompt:
                    tasks = await api.list_coding_tasks(status="all")
                    await events.emit("Coding tasks", format_coding_tasks(tasks, active_only=False), "bright_magenta")
                    continue
                if command == "/coding-tasks":
                    tasks = await api.list_coding_tasks(status="all")
                    await events.emit("Coding tasks", format_coding_tasks(tasks, active_only=False), "bright_magenta")
                    continue
                if ui is not None:
                    ui.set_status("starting coding task...", "busy")
                result = await api.start_coding_task(task_prompt)
                task = result.get("task") or {}
                await events.emit("Coding task", format_coding_task(task), "bright_magenta")
            except Exception as exc:
                if ui is not None:
                    ui.set_status("coding command failed", "error")
                await events.emit("Coding command failed", str(exc), "red")
            continue

        if command == "/coding-continue":
            message = prompt.partition(" ")[2].strip()
            if not message:
                await events.emit(
                    "Coding continue",
                    "Usage: `/coding-continue <message>`\nOr: `/coding-continue <task_id> <message>`",
                    "yellow",
                )
                continue
            task_id = ""
            if words[1:2] and words[1].startswith("code_") and len(words) >= 3:
                task_id = words[1]
                message = prompt.split(None, 2)[2]
            try:
                if ui is not None:
                    ui.set_status("continuing coding task...", "busy")
                if task_id:
                    result = await api.reply_to_coding_task(task_id, message)
                else:
                    result = await api.reply_to_latest_coding_task(message)
                await events.emit("Coding continue", format_coding_task(result.get("task") or {}), "bright_magenta")
            except Exception as exc:
                if ui is not None:
                    ui.set_status("coding continue failed", "error")
                await events.emit("Coding continue failed", str(exc), "red")
            continue

        if command in {"/coding-agents", "/coding-config"}:
            try:
                config = await api.list_coding_agents()
                await events.emit("Coding agents", format_coding_agent_config(config), "bright_magenta")
                if ui is not None and hasattr(ui, "refresh_header"):
                    ui.refresh_header()
            except Exception as exc:
                await events.emit("Coding agents failed", str(exc), "red")
            continue

        if command == "/coding-use":
            provider = words[1] if len(words) > 1 else ""
            if not provider:
                await events.emit("Coding agent", "Usage: `/coding-use <provider>`\nExample: `/coding-use codex`", "yellow")
                continue
            try:
                config = await api.set_coding_agent(provider)
                await events.emit("Coding agent selected", format_coding_agent_config(config), "bright_magenta")
                if ui is not None and hasattr(ui, "refresh_header"):
                    ui.refresh_header()
            except Exception as exc:
                await events.emit("Coding agent select failed", str(exc), "red")
            continue

        if command == "/coding-workspace":
            workspace_path = prompt.partition(" ")[2].strip()
            if not workspace_path:
                await events.emit("Coding workspace", "Usage: `/coding-workspace <path>`", "yellow")
                continue
            try:
                config = await api.set_coding_workspace(workspace_path)
                await events.emit("Coding workspace updated", format_coding_agent_config(config), "bright_magenta")
                if ui is not None and hasattr(ui, "refresh_header"):
                    ui.refresh_header()
            except Exception as exc:
                await events.emit("Coding workspace failed", str(exc), "red")
            continue

        if command == "/coding-network":
            args = words[1:]
            try:
                if not args:
                    config = await api.list_coding_agents()
                    await events.emit("Coding network", format_coding_agent_config(config), "bright_magenta")
                    continue
                if args[0].startswith("code_"):
                    if len(args) < 2:
                        await events.emit("Coding network", "Usage: `/coding-network <task_id> on|off`", "yellow")
                        continue
                    enabled = parse_on_off(args[1])
                    if enabled is None:
                        await events.emit("Coding network", "Usage: `/coding-network <task_id> on|off`", "yellow")
                        continue
                    result = await api.set_coding_task_network(args[0], enabled)
                    await events.emit("Coding network updated", format_coding_task(result.get("task") or {}), "bright_magenta")
                    continue
                enabled = parse_on_off(args[0])
                if enabled is None:
                    await events.emit(
                        "Coding network",
                        "Usage: `/coding-network on|off`\nOr: `/coding-network <task_id> on|off`",
                        "yellow",
                    )
                    continue
                config = await api.set_coding_network(enabled)
                await events.emit("Coding network updated", format_coding_agent_config(config), "bright_magenta")
                if ui is not None and hasattr(ui, "refresh_header"):
                    ui.refresh_header()
            except Exception as exc:
                await events.emit("Coding network failed", str(exc), "red")
            continue

        if command == "/coding-allow-network":
            if len(words) < 2:
                await events.emit("Coding network", "Usage: `/coding-allow-network <task_id>`", "yellow")
                continue
            try:
                result = await api.set_coding_task_network(words[1], True)
                await events.emit("Coding network updated", format_coding_task(result.get("task") or {}), "bright_magenta")
            except Exception as exc:
                await events.emit("Coding network failed", str(exc), "red")
            continue

        if command == "/coding-status":
            if len(words) < 2:
                await events.emit("Coding status", "Usage: `/coding-status <task_id>`", "yellow")
                continue
            try:
                task = await api.get_coding_task(words[1])
                await events.emit("Coding status", format_coding_task(task), "bright_magenta")
            except Exception as exc:
                await events.emit("Coding status failed", str(exc), "red")
            continue

        if command == "/coding-log":
            if len(words) < 2:
                await events.emit("Coding log", "Usage: `/coding-log <task_id>`", "yellow")
                continue
            try:
                logs = await api.get_coding_logs(words[1], limit=50)
                await events.emit("Coding log", format_coding_logs(logs), "bright_magenta")
            except Exception as exc:
                await events.emit("Coding log failed", str(exc), "red")
            continue

        if command == "/coding-reply":
            if len(words) < 3:
                await events.emit(
                    "Coding reply",
                    "Usage: `/coding-reply <task_id> <message>`",
                    "yellow",
                )
                continue
            task_id = words[1]
            message = prompt.split(None, 2)[2]
            try:
                result = await api.reply_to_coding_task(task_id, message)
                await events.emit("Coding reply", format_coding_task(result.get("task") or {}), "bright_magenta")
            except Exception as exc:
                await events.emit("Coding reply failed", str(exc), "red")
            continue

        if command == "/coding-cancel":
            if len(words) < 2:
                await events.emit("Coding cancel", "Usage: `/coding-cancel <task_id>`", "yellow")
                continue
            try:
                result = await api.cancel_coding_task(words[1])
                await events.emit("Coding cancelled", format_coding_task(result.get("task") or {}), "bright_magenta")
            except Exception as exc:
                await events.emit("Coding cancel failed", str(exc), "red")
            continue

        if command in {"/coding-delete", "/coding-rm"}:
            if len(words) < 2:
                await events.emit("Coding delete", "Usage: `/coding-delete <task_id>`", "yellow")
                continue
            try:
                result = await api.delete_coding_task(words[1])
                await events.emit(
                    "Coding task deleted",
                    result.get("message") or f"Deleted coding task {words[1]}.",
                    "bright_magenta",
                )
            except Exception as exc:
                await events.emit("Coding delete failed", str(exc), "red")
            continue

        if command in {"/coding-clear", "/coding-clear-history"}:
            try:
                result = await api.clear_coding_task_history()
                await events.emit(
                    "Coding history cleared",
                    (
                        f"Deleted {result.get('deleted', 0)} coding task(s).\n"
                        f"Preserved {result.get('preserved', 0)} active task(s)."
                    ),
                    "bright_magenta",
                )
            except Exception as exc:
                await events.emit("Coding clear failed", str(exc), "red")
            continue

        if command == "/connect":
            channel = words[1] if len(words) > 1 else ""
            try:
                if ui is not None:
                    ui.set_status(f"connecting {channel or 'channel'}...", "busy")
                result = await channel_manager.connect(channel)
                await events.emit("Channel connected", result, "green")
            except Exception as exc:
                if ui is not None:
                    ui.set_status("channel connection failed", "error")
                await events.emit("Channel connection failed", str(exc), "red")
            continue

        if command in {"set-default", "/set-default"}:
            if len(words) < 2 or words[1].lower() != "start-channel":
                await events.emit(
                    "Default command",
                    "Usage: set-default start-channel <channel...>\nExample: set-default start-channel telegram\nClear: set-default start-channel none",
                    "yellow",
                )
                continue

            requested_channels = words[2:]
            if not requested_channels:
                await events.emit(
                    "Default command",
                    "Usage: set-default start-channel <channel...>\nExample: set-default start-channel telegram",
                    "yellow",
                )
                continue

            if len(requested_channels) == 1 and normalize_channel_type(requested_channels[0]) in {"none", "clear", "off", "disable", "disabled"}:
                channels = []
            else:
                channels = normalize_channel_list(requested_channels)

            config = save_cli_config({"startup_channels": channels})
            if ui is not None:
                ui.set_default_start_channels(config["startup_channels"])
                ui.set_status("defaults updated", "ready")
            await events.emit("Defaults updated", format_set_default_response(config), "green")
            continue

        try:
            if ui is not None:
                ui.set_status("queueing request...", "busy")
            job = await api.create_chat_job(prompt, channel_type="cli")
            job_id = str(job.get("id") or "")
            if not job_id:
                raise RuntimeError("API did not return a job id.")

            completed_job = await poll_job_until_complete(api, job_id, ui)
            if completed_job.get("status") == "failed":
                raise RuntimeError(str(completed_job.get("error") or "agent job failed"))
        except Exception as exc:
            if ui is not None:
                ui.set_status("request failed", "error")
            await events.emit("CLI request failed", str(exc), "red")
        else:
            if ui is not None:
                ui.set_status("ready", "ready")


async def api_event_loop(
    api: QuarqApiClient,
    events: EventLog,
    stop_event: asyncio.Event,
) -> None:
    last_event_id = 0
    while not stop_event.is_set():
        try:
            api_events = await api.events_since(last_event_id)
            for event in api_events:
                last_event_id = max(last_event_id, int(event.get("id") or 0))
                if events.ui is not None:
                    update_status_from_api_event(events.ui, event)
                await events.emit_api_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await events.emit("Event stream paused", str(exc), "yellow")
            await asyncio.sleep(2)

        await asyncio.sleep(EVENT_POLL_INTERVAL)


def format_status(data: dict[str, Any]) -> str:
    rows = []
    configured_channels = []
    if data.get("telegram_configured"):
        configured_channels.append("telegram")
    rows.append(f"configured_channels: {channel_summary(configured_channels)}")

    skipped_keys = {
        "telegram_allowed_users_configured",
        "telegram_configured",
        "telegram_webhook_path",
        "coding_agent",
    }
    coding_agent = data.get("coding_agent") or {}
    if coding_agent:
        rows.append(
            f"coding_agent: {coding_agent.get('default_provider_label')} "
            f"(`{coding_agent.get('default_provider')}`)"
        )
        rows.append(f"coding_workspace: {coding_agent.get('workspace_root')}")
        rows.append(f"coding_network: {'on' if coding_agent.get('network_access') else 'off'}")
    for key, value in sorted(data.items()):
        if key in skipped_keys:
            continue
        rows.append(f"{key}: {value}")
    return "\n".join(rows)


def update_status_from_api_event(ui: TerminalUi, event: dict[str, Any]) -> None:
    title = str(event.get("title") or "").lower()
    kind = str(event.get("kind") or "").lower()
    data = event.get("data") or {}
    channel = str(data.get("channel") or "").lower()

    if title in {"chat request", "telegram inbound"} or (kind == "request" and channel):
        ui.set_status("waiting for response...", "busy")
    elif kind == "coding":
        status = str(data.get("status") or "").strip()
        if status == "completed":
            ui.set_status("coding complete", "ready")
        elif status == "failed":
            ui.set_status("coding failed", "error")
        elif status == "waiting_user":
            ui.set_status("coding waiting for reply", "notice")
        else:
            ui.set_status("coding...", "busy")
        if data.get("status") == "configured" and hasattr(ui, "refresh_header"):
            ui.refresh_header()
    elif kind == "job":
        tool_name = str(data.get("tool_name") or "").strip()
        message = str(event.get("message") or "").strip()
        if tool_name:
            ui.set_status(f"using tool: {tool_name}", "busy")
        elif message:
            ui.set_status(message.lower(), "busy")
        else:
            ui.set_status("working...", "busy")
    elif title in {"chat response", "telegram response", "command handled"}:
        ui.set_status("ready", "ready")
        if hasattr(ui, "refresh_header"):
            ui.refresh_header()
    elif "error" in title or kind == "error":
        ui.set_status("error - check transcript", "error")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Quarq local control console.")
    parser.add_argument("--host", default=os.getenv("QUARQ_API_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int(os.getenv("QUARQ_API_PORT", DEFAULT_PORT)))
    parser.add_argument("--no-server", action="store_true", help="Connect to an already running API instead of starting main:app.")
    parser.add_argument("--api-base", default=None, help="Override API base URL, for example http://127.0.0.1:8000.")
    parser.add_argument("--server-log-level", default=os.getenv("QUARQ_API_LOG_LEVEL", "warning"))
    parser.add_argument("--no-tunnel", action="store_true", help="Disable public tunnel setup for channel connections.")
    parser.add_argument("--cloudflared", default=os.getenv("CLOUDFLARED_BIN", "cloudflared"), help="Path or command name for cloudflared.")
    return parser


async def amain() -> int:
    args = build_parser().parse_args()
    console = Console()
    api_base = args.api_base or f"http://{args.host}:{args.port}"

    ui = build_terminal_ui(api_base)
    if ui is None:
        print_header(console, api_base)
    events = EventLog(console, ui)
    api = QuarqApiClient(api_base, events)
    server = None
    channel_manager = ChannelManager(
        api_base,
        events,
        ui,
        args.cloudflared,
        tunnel_enabled=not args.no_tunnel,
    )
    stop_event = asyncio.Event()
    tasks: list[asyncio.Task] = []
    ui_task: asyncio.Task | None = None

    def request_shutdown(*_: Any) -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_shutdown)
        except NotImplementedError:
            pass

    try:
        if ui is not None:
            ui_task = asyncio.create_task(ui.run())

        if not args.no_server:
            if ui is not None:
                ui.set_status("starting api server...", "busy")
            await events.emit("API server", f"Starting main:app on {args.host}:{args.port}", "yellow")
            server = start_api_server(args.host, args.port, args.server_log_level)

        if ui is None:
            await events.emit(
                "Basic input mode",
                "Install textual from requirements.txt for the fixed bottom input row and native transcript scrolling.",
                "yellow",
            )

        if ui is not None:
            ui.set_status("waiting for api...", "busy")
        await wait_for_api(api, timeout=45)
        status = await api.health()
        await events.emit("API ready", format_status(status), "green")
        if ui is not None:
            ui.set_status("api ready", "ready")

        startup_channels = load_cli_config().get("startup_channels", [])
        if startup_channels and args.no_tunnel:
            await events.emit(
                "Startup channels skipped",
                "Tunnels are disabled for this CLI run.",
                "yellow",
            )
        channels_to_start = [] if args.no_tunnel else startup_channels
        for channel in channels_to_start:
            try:
                if ui is not None:
                    ui.set_status(f"connecting {channel}...", "busy")
                await events.emit(
                    "Startup channel",
                    f"Connecting {channel} from saved startup defaults.",
                    "yellow",
                )
                result = await channel_manager.connect(channel)
                await events.emit("Channel connected", result, "green")
            except Exception as exc:
                if ui is not None:
                    ui.set_status(f"{channel} connection failed", "error")
                await events.emit("Startup channel failed", str(exc), "red")

        if ui is not None:
            ui.set_status("ready", "ready")
        tasks.append(asyncio.create_task(api_event_loop(api, events, stop_event)))
        tasks.append(asyncio.create_task(cli_input_loop(api, events, console, ui, channel_manager, stop_event)))
        await stop_event.wait()
        return 0

    finally:
        if ui is not None:
            ui.set_status("shutting down...", "busy")
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await api.close()
        await channel_manager.close()
        if server is not None:
            await stop_api_server(server)
        if ui is None:
            await events.emit("Shutdown", "Control console stopped.", "yellow")
        if ui is not None:
            ui.close()
        if ui_task is not None:
            await asyncio.gather(ui_task, return_exceptions=True)


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        raise SystemExit(0)


if __name__ == "__main__":
    main()
