"""Minimal ClickHouse HTTP client used by the generation forecast pipeline."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ClickHouseQueryError(RuntimeError):
    """Raised when ClickHouse returns an error or the request fails."""


class ClickHouseClient:
    def __init__(
        self,
        *,
        url: str,
        database: str,
        username: str,
        password: str,
        timeout: int = 120,
    ) -> None:
        self.url = url.rstrip("/")
        self.database = database
        self.username = username
        self.password = password
        self.timeout = timeout

    def query_json_each_row(self, query: str) -> list[dict[str, Any]]:
        request = Request(
            f"{self.url}?{urlencode({'database': self.database, 'user': self.username, 'password': self.password, 'date_time_input_format': 'best_effort'})}",
            data=f"{query.rstrip()}\nFORMAT JSONEachRow".encode("utf-8"),
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except (HTTPError, URLError) as exc:
            raise ClickHouseQueryError(str(exc)) from exc

        rows = []
        for line in body.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    def command(self, query: str) -> None:
        request = Request(
            f"{self.url}?{urlencode({'database': self.database, 'user': self.username, 'password': self.password, 'date_time_input_format': 'best_effort'})}",
            data=query.rstrip().encode("utf-8"),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response.read()
        except (HTTPError, URLError) as exc:
            raise ClickHouseQueryError(str(exc)) from exc

    def insert_json_each_row(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return

        payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        request = Request(
            f"{self.url}?{urlencode({'database': self.database, 'user': self.username, 'password': self.password, 'date_time_input_format': 'best_effort'})}",
            data=f"INSERT INTO {table} FORMAT JSONEachRow\n{payload}".encode("utf-8"),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response.read()
        except (HTTPError, URLError) as exc:
            raise ClickHouseQueryError(str(exc)) from exc


@lru_cache
def get_clickhouse_client(
    url: str | None = None,
    database: str | None = None,
    username: str | None = None,
    password: str | None = None,
    timeout: int | None = None,
) -> ClickHouseClient:
    return ClickHouseClient(
        url=url or os.getenv("GENERATION_CLICKHOUSE_URL", "http://localhost:8123"),
        database=database or os.getenv("GENERATION_CLICKHOUSE_DATABASE", "kegoc"),
        username=username or os.getenv("GENERATION_CLICKHOUSE_USERNAME", "default"),
        password=password or os.getenv("GENERATION_CLICKHOUSE_PASSWORD", ""),
        timeout=timeout or int(os.getenv("GENERATION_CLICKHOUSE_TIMEOUT_SECONDS", "120")),
    )
