"""Client for the private API of the Telemost web client (telemost.360.yandex.ru).

Two channels share one Yandex session. The user pastes its Cookie header into the env file:
- HTTP `POST messenger.360.yandex.ru/api/` takes the form field `request={"method", "params"}`.
- The Xiva WebSocket at push.yandex.ru carries `whoami`, `history` and other fanout calls.
  A frame is msgpack `1, [service_index, req_id, method]`, a 12-byte header, then JSON.
- Attachments come from `GET <files host>/file_shortterm/<FileInfo.Id2>`, which redirects to a signed storage URL.
The protocol is reverse-engineered from web client 211.3.0. Yandex can change it without notice.
"""

import asyncio
import itertools
import json
import os
import re
import secrets
import string
import uuid
from pathlib import Path
from urllib.parse import urlencode

import httpx
import msgpack
import websockets

ORIGIN = "https://telemost.360.yandex.ru"
API_URL = "https://messenger.360.yandex.ru/api/"
XIVA_URL = "wss://push.yandex.ru/v2/subscribe/websocket"
XIVA_SERVICE = "messenger-prod:version5*common+version5*main"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)
API_HEADERS = {
    "X-Application-Id": "Yamb-web",
    "X-Origin-Service-ID": "27",
    "X-Version": "5",
    "Origin": ORIGIN,
    "Referer": ORIGIN + "/",
    "User-Agent": USER_AGENT,
}
# The business host is tried first, then the consumer one. TELEMOST_FILES_HOST pins a single host.
FILE_HOSTS = (
    [os.environ["TELEMOST_FILES_HOST"]] if os.environ.get("TELEMOST_FILES_HOST")
    else ["files.messenger.360.yandex.ru", "files.messenger.yandex.ru"]
)
# FileInfo.Id2 is `<bucket>/<uuid>`. It goes into the URL path unescaped, so only plain segments pass.
FILE_ID = re.compile(r"[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*")
# The web client writes 5 into byte 0 of this header and leaves the rest empty.
PAYLOAD_HEADER = b"\x05" + bytes(11)
DATA, PROXY_STATUS = 1, 2


class TelemostError(RuntimeError):
    pass


ENV_FILE = Path(os.environ.get("TELEMOST_ENV", "~/.config/telemost-mcp/.env")).expanduser()


def cookie_header() -> str:
    """Read TELEMOST_COOKIE from the env file on every call, so a re-pasted cookie applies without a restart."""
    lines = ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else []
    values = dict(line.split("=", 1) for line in lines if line.startswith("TELEMOST_COOKIE="))
    cookie = values.get("TELEMOST_COOKIE", os.environ.get("TELEMOST_COOKIE", "")).strip().strip("'\"")
    if "Session_id=" not in cookie:
        raise TelemostError(
            f"No Yandex session cookie. Copy the Cookie header of a messenger.360.yandex.ru/api/ request "
            f"from DevTools into {ENV_FILE} as TELEMOST_COOKIE=..."
        )
    return cookie


class Xiva:
    """One Xiva WebSocket. It matches each response to its request by req_id."""

    def __init__(self, cookie: str, uid: int):
        self.cookie = cookie
        self.uid = uid
        self.pending: dict[int, asyncio.Future] = {}
        self.req_ids = itertools.count(1)

    async def connect(self) -> None:
        session = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(19))
        query = urlencode(
            {"service": XIVA_SERVICE, "session": session, "client": "web_main", "user": self.uid}, safe="*"
        )
        self.ws = await websockets.connect(
            f"{XIVA_URL}?{query}",
            additional_headers={"Cookie": self.cookie, "Origin": ORIGIN, "User-Agent": USER_AGENT},
            max_size=None,
        )
        # The server confirms the subscription before it accepts requests.
        async with asyncio.timeout(10):
            async for frame in self.ws:
                operation = json.loads(frame).get("operation") if isinstance(frame, str) else None
                if operation == "subscribed":
                    break
                if operation == "xivaws-error":
                    raise TelemostError(f"Xiva refused the connection: {frame}")
        self.reader = asyncio.create_task(self.read())

    @property
    def alive(self) -> bool:
        return not self.reader.done()

    async def read(self) -> None:
        try:
            async for frame in self.ws:
                if isinstance(frame, str):
                    continue  # Pings and service operations.
                unpacker = msgpack.Unpacker(raw=False)
                unpacker.feed(frame)
                kind, header = unpacker.unpack(), unpacker.unpack()
                if kind == DATA:
                    self.resolve(header[1], json.loads(frame[unpacker.tell() + len(PAYLOAD_HEADER):]))
                elif kind == PROXY_STATUS:
                    self.resolve(header[0], TelemostError(f"Xiva proxy error code {header[1]}"))
        finally:
            for req_id in list(self.pending):
                self.resolve(req_id, TelemostError("Xiva connection closed"))

    def resolve(self, req_id: int, result) -> None:
        future = self.pending.pop(req_id, None)
        if future is None or future.done():
            return
        if isinstance(result, Exception):
            future.set_exception(result)
        else:
            future.set_result(result)

    async def call(self, method: str, body: dict, timeout: float = 30) -> dict:
        req_id = next(self.req_ids)
        future = asyncio.get_running_loop().create_future()
        self.pending[req_id] = future
        payload = json.dumps({"RequestId": str(uuid.uuid4()), **body}).encode()
        await self.ws.send(msgpack.packb(DATA) + msgpack.packb([0, req_id, method]) + PAYLOAD_HEADER + payload)
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self.pending.pop(req_id, None)


class Telemost:
    """Session facade: HTTP API calls and a lazily (re)connected Xiva socket."""

    def __init__(self):
        self.http = httpx.AsyncClient(headers=API_HEADERS, timeout=30)
        self.me: dict | None = None
        self.xiva: Xiva | None = None
        self.lock = asyncio.Lock()

    async def api(self, method: str, **params) -> dict:
        headers = {"X-Request-Id": str(uuid.uuid4()), "X-Request-Attempt": "0", "Cookie": cookie_header()}
        if self.me:
            headers |= {"X-Uid": str(self.me["uid"]), "X-Ya-Organization-Id": str(self.me["organization_id"])}
        request = json.dumps({"method": method, "params": params})
        response = await self.http.post(API_URL, files={"request": (None, request)}, headers=headers)
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "ok":
            raise TelemostError(f"{method} failed: {body}")
        return body["data"]

    async def download(self, file_id: str, dest: Path) -> str:
        """Write the attachment to `dest` and return its Content-Type."""
        if not FILE_ID.fullmatch(file_id) or ".." in file_id.split("/"):
            raise TelemostError(f"Not an attachment id: {file_id!r}. Pass `id` from a message's `files`.")
        headers = {"Cookie": cookie_header(), "Origin": ORIGIN, "Referer": ORIGIN + "/", "User-Agent": USER_AGENT}
        errors = []
        for host in FILE_HOSTS:
            try:
                response = await self.http.get(f"https://{host}/file_shortterm/{file_id}", params={"attach": "true"}, headers=headers)
            except httpx.HTTPError as error:
                errors.append(f"{host}: {type(error).__name__}")
                continue
            if response.is_redirect:
                # The storage URL is signed. The session cookie must not follow it there.
                url, headers = response.headers["Location"], {"User-Agent": USER_AGENT}
            elif response.is_success:
                dest.write_bytes(response.content)
                return response.headers.get("Content-Type", "")
            else:
                errors.append(f"{host}: HTTP {response.status_code}")
                continue
            async with httpx.AsyncClient(timeout=300, follow_redirects=True) as storage:
                async with storage.stream("GET", url, headers=headers) as stored:
                    stored.raise_for_status()
                    with dest.open("wb") as out:
                        async for chunk in stored.aiter_bytes():
                            out.write(chunk)
                    return stored.headers.get("Content-Type", "")
        raise TelemostError(f"Could not download {file_id}: {'; '.join(errors)}")

    async def fanout(self, method: str, body: dict) -> dict:
        async with self.lock:
            if self.xiva is None or not self.xiva.alive:
                await self.login()
        return await self.xiva.call(method, body)

    async def login(self) -> None:
        cookie = cookie_header()
        # Session_id lists the logged-in accounts as `|<uid>.`; the first one is the default account.
        self.xiva = Xiva(cookie, int(re.search(r"Session_id=[^;]*?\|(\d+)\.", cookie).group(1)))
        await self.xiva.connect()
        info = (await self.xiva.call("whoami", {}))["UserInfo"]
        organizations = info.get("OrganizationInfos") or [{}]
        self.me = {
            "uid": info["Uid"],
            "guid": info["Guid"],
            "name": info.get("DisplayName"),
            "email": info.get("Email"),
            "organization_id": organizations[0].get("OrganizationId", 0),
            "organization": organizations[0].get("OrganizationName"),
        }
