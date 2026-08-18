"""토스증권 Open API 기반 KOSPI 대규모 수집·이미지·CNN 추론 파이프라인.

계좌·잔고·보유종목·주문 기능은 의도적으로 구현하지 않았습니다. 아래 허용 목록의
종목/현재가/일봉 API만 호출합니다. Google Colab에서 500종목을 처리하다 중단되더라도
종목별 체크포인트와 원본 캐시를 이용해 이어서 실행할 수 있도록 구성했습니다.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import matplotlib

# FastAPI의 worker thread나 GUI가 없는 Docker/Colab에서도 안전하게 렌더링합니다.
# pyplot을 import하기 전에 지정해야 실제 백엔드가 확실히 고정됩니다.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests
from matplotlib.collections import LineCollection
from PIL import Image


BASE_URL = "https://openapi.tossinvest.com"
MARKET_ENDPOINTS = frozenset(
    {
        "/api/v1/stocks/all",
        "/api/v1/stocks",
        "/api/v1/prices",
        "/api/v1/candles",
    }
)
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]


@dataclass(frozen=True)
class PipelineConfig:
    """수집 및 차트 생성 설정.

    ``top_n``만 500/400/300 중 원하는 값으로 바꾸면 처리 기업 수가 달라집니다.
    6개월·12개월 구간 안의 실제 일봉 전체를 연속 OHLC 100개로 집계합니다.
    """

    start_date: str = "2016-01-01"
    end_date: str = "2026-12-31"
    top_n: int = 500
    target_candles: int = 100
    period_months: tuple[int, ...] = (6, 12)
    adjusted: bool = True
    page_size: int = 200
    request_pause_seconds: float = 0.23
    max_retries: int = 6
    figsize_inches: tuple[int, int] = (3, 3)
    dpi: int = 100
    connect_price_gaps: bool = True
    gap_connector_linewidth: float = 0.55


class TossAPIError(RuntimeError):
    """Sanitized API error that never includes credentials or request headers."""


class TossMarketClient:
    """Minimal client locked to auth and four non-account market endpoints."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        max_retries: int = 6,
        pause_seconds: float = 0.23,
    ) -> None:
        if not client_id or not client_secret:
            raise ValueError("TOSS_CLIENT_ID / TOSS_CLIENT_SECRET 값이 비어 있습니다.")
        self._client_id = client_id
        self._client_secret = client_secret
        self._max_retries = max_retries
        self._pause_seconds = pause_seconds
        self._session = requests.Session()
        self._access_token: str | None = None
        self._token_expires_at = 0.0

    def _issue_token(self) -> None:
        response = self._session.post(
            f"{BASE_URL}/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self._client_id,
                "client_secret": self._client_secret,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        if response.status_code != 200:
            if response.status_code == 403:
                raise TossAPIError(
                    "토큰 발급이 403으로 거절되었습니다. 현재 Colab 외부 IP를 "
                    "토스증권 WTS > 설정 > Open API > 허용 IP에 등록하세요. "
                    f"서버 응답: {self._error_message(response)}"
                )
            raise TossAPIError(
                f"토큰 발급 실패(HTTP {response.status_code}). "
                "Client ID/Secret과 Open API 활성 상태를 확인하세요."
            )
        payload = response.json()
        self._access_token = payload["access_token"]
        expires_in = int(payload.get("expires_in", 86400))
        self._token_expires_at = time.time() + max(60, expires_in - 60)

    def _ensure_token(self) -> None:
        if self._access_token is None or time.time() >= self._token_expires_at:
            self._issue_token()

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error", payload)
            if isinstance(error, dict):
                code = error.get("code") or error.get("error") or "unknown"
                message = error.get("message") or error.get("error_description") or ""
                return f"{code}: {message}".strip()
        except Exception:
            pass
        return "응답 본문을 해석할 수 없습니다."

    def get(self, path: str, *, params: dict | None = None) -> dict:
        if path not in MARKET_ENDPOINTS:
            raise ValueError(f"허용되지 않은 endpoint입니다: {path}")

        token_refreshed = False
        for attempt in range(self._max_retries):
            self._ensure_token()
            response = self._session.get(
                f"{BASE_URL}{path}",
                params=params,
                headers={"Authorization": f"Bearer {self._access_token}"},
                timeout=30,
            )

            if response.status_code == 200:
                time.sleep(self._pause_seconds)
                return response.json()

            if response.status_code == 401 and not token_refreshed:
                self._access_token = None
                token_refreshed = True
                continue

            if response.status_code == 429:
                delay = float(
                    response.headers.get("Retry-After")
                    or response.headers.get("X-RateLimit-Reset")
                    or min(2 ** attempt, 8)
                )
                time.sleep(max(delay, 0.25))
                continue

            if response.status_code >= 500 and attempt + 1 < self._max_retries:
                time.sleep(min(2 ** attempt, 8))
                continue

            raise TossAPIError(
                f"GET {path} 실패(HTTP {response.status_code}) - "
                f"{self._error_message(response)}"
            )

        raise TossAPIError(f"GET {path} 재시도 한도를 초과했습니다.")


def _chunks(values: list[str], size: int = 200) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _normalize_symbol(value: object) -> str:
    return str(value).strip().zfill(6)


def select_current_kospi_top_n(
    client: TossMarketClient,
    config: PipelineConfig,
) -> pd.DataFrame:
    """현재 KOSPI 보통주를 시가총액 추정치 순으로 ``top_n``개 선정합니다.

    토스가 시가총액 순위를 직접 주지는 않으므로 발행주식수 × 현재가로 계산합니다.
    ETF·ETN·우선주·코스닥은 API 필터와 아래 검증식에서 제외됩니다.
    """
    listed = client.get(
        "/api/v1/stocks/all",
        params={
            "market": "KOSPI",
            "status": "ACTIVE",
            "securityType": "STOCK",
            "commonShare": "true",
        },
    ).get("result", [])
    if not listed:
        raise TossAPIError("KOSPI 종목 목록이 비어 있습니다.")

    listed_df = pd.DataFrame(listed)
    listed_df["symbol"] = listed_df["symbol"].map(_normalize_symbol)
    symbols = listed_df["symbol"].drop_duplicates().tolist()

    details: list[dict] = []
    prices: list[dict] = []
    for group in _chunks(symbols, 200):
        symbol_csv = ",".join(group)
        details.extend(
            client.get("/api/v1/stocks", params={"symbols": symbol_csv}).get(
                "result", []
            )
        )
        prices.extend(
            client.get("/api/v1/prices", params={"symbols": symbol_csv}).get(
                "result", []
            )
        )

    details_df = pd.DataFrame(details)
    prices_df = pd.DataFrame(prices)
    if details_df.empty or prices_df.empty:
        raise TossAPIError("종목 상세 또는 현재가 응답이 비어 있습니다.")

    for frame in (details_df, prices_df):
        frame["symbol"] = frame["symbol"].map(_normalize_symbol)

    price_keep = prices_df[["symbol", "lastPrice"]].drop_duplicates("symbol")
    universe = details_df.merge(price_keep, on="symbol", how="inner")
    universe["sharesOutstanding"] = pd.to_numeric(
        universe["sharesOutstanding"], errors="coerce"
    )
    universe["lastPrice"] = pd.to_numeric(universe["lastPrice"], errors="coerce")

    mask = (
        universe["market"].eq("KOSPI")
        & universe["securityType"].eq("STOCK")
        & universe["isCommonShare"].eq(True)
        & universe["status"].eq("ACTIVE")
        & universe["sharesOutstanding"].gt(0)
        & universe["lastPrice"].gt(0)
    )
    universe = universe.loc[mask].copy()
    universe["marketCap"] = universe["sharesOutstanding"] * universe["lastPrice"]
    universe = (
        universe.sort_values(["marketCap", "symbol"], ascending=[False, True])
        .drop_duplicates("symbol")
        .head(config.top_n)
        .reset_index(drop=True)
    )
    universe.insert(0, "rank", np.arange(1, len(universe) + 1))

    if len(universe) != config.top_n:
        raise TossAPIError(
            f"시가총액 계산 가능한 KOSPI 보통주가 {len(universe)}개뿐입니다."
        )

    columns = [
        "rank",
        "symbol",
        "name",
        "market",
        "listDate",
        "sharesOutstanding",
        "lastPrice",
        "marketCap",
    ]
    return universe[[column for column in columns if column in universe.columns]]


def _effective_end_date(value: str) -> pd.Timestamp:
    configured = pd.Timestamp(value).normalize()
    today_kst = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    return min(configured, today_kst)


def _clean_daily_candles(candles: list[dict]) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame(columns=OHLC_COLUMNS)

    frame = pd.DataFrame(candles)
    required = {
        "timestamp",
        "openPrice",
        "highPrice",
        "lowPrice",
        "closePrice",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise TossAPIError(f"캔들 응답 필드 누락: {sorted(missing)}")

    index = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame.index = index.dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.normalize()
    frame = frame.rename(
        columns={
            "openPrice": "Open",
            "highPrice": "High",
            "lowPrice": "Low",
            "closePrice": "Close",
            "volume": "Volume",
        }
    )
    numeric_columns = OHLC_COLUMNS + (["Volume"] if "Volume" in frame else [])
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    frame = frame.dropna(subset=OHLC_COLUMNS)
    frame = frame[~frame.index.isna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()

    valid = (
        frame["High"].ge(frame[["Open", "Close", "Low"]].max(axis=1))
        & frame["Low"].le(frame[["Open", "Close", "High"]].min(axis=1))
        & frame["Low"].gt(0)
    )
    return frame.loc[valid]


def fetch_daily_history(
    client: TossMarketClient,
    symbol: str,
    start_date: str,
    end_date: str,
    *,
    adjusted: bool = True,
    page_size: int = 200,
) -> pd.DataFrame:
    """Page backwards through Toss 1d candles, then keep the requested dates."""
    start = pd.Timestamp(start_date).normalize()
    end = _effective_end_date(end_date)
    before = f"{end.date().isoformat()}T23:59:59+09:00"
    all_rows: list[dict] = []
    seen_timestamps: set[str] = set()
    previous_oldest: pd.Timestamp | None = None

    while True:
        payload = client.get(
            "/api/v1/candles",
            params={
                "symbol": _normalize_symbol(symbol),
                "interval": "1d",
                "count": min(page_size, 200),
                "before": before,
                "adjusted": str(adjusted).lower(),
            },
        )
        result = payload.get("result") or {}
        page = result.get("candles") or []
        if not page:
            break

        new_rows = []
        for row in page:
            timestamp = row.get("timestamp")
            if timestamp and timestamp not in seen_timestamps:
                seen_timestamps.add(timestamp)
                new_rows.append(row)
        all_rows.extend(new_rows)

        parsed_page = _clean_daily_candles(page)
        if parsed_page.empty:
            break
        oldest = parsed_page.index.min()
        if oldest < start:
            break

        next_before = result.get("nextBefore")
        if not next_before:
            break
        if previous_oldest is not None and oldest >= previous_oldest:
            raise TossAPIError(
                f"{symbol}: 캔들 페이지가 과거로 진행되지 않습니다(nextBefore 확인 필요)."
            )
        previous_oldest = oldest
        before = next_before

    history = _clean_daily_candles(all_rows)
    history = history.loc[(history.index >= start) & (history.index <= end)]
    history.index.name = "Date"
    return history


def _monochrome_style() -> dict:
    colors = mpf.make_marketcolors(
        up="black",
        down="black",
        edge="black",
        wick="black",
        ohlc="black",
        volume="white",
    )
    return mpf.make_mpf_style(
        marketcolors=colors,
        facecolor="white",
        figcolor="white",
        gridcolor="white",
        y_on_right=False,
        rc={
            "axes.grid": False,
            "axes.edgecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        },
    )


def render_100_candles(
    frame: pd.DataFrame,
    output_path: str | Path,
    config: PipelineConfig,
) -> dict:
    if len(frame) != config.target_candles:
        raise ValueError(
            f"이미지 입력은 {config.target_candles}개 캔들이어야 합니다: {len(frame)}개"
        )
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("캔들 날짜가 오름차순/고유값이 아닙니다.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp.png")
    figure, axes = mpf.plot(
        frame[OHLC_COLUMNS],
        type="candle",
        style=_monochrome_style(),
        volume=False,
        axisoff=True,
        show_nontrading=False,
        figsize=config.figsize_inches,
        tight_layout=False,
        returnfig=True,
        warn_too_much_data=1000,
    )
    for axis in axes:
        axis.set_axis_off()
        axis.grid(False)
    if config.connect_price_gaps:
        segments = [
            [
                (position - 1, float(frame["Close"].iloc[position - 1])),
                (position, float(frame["Open"].iloc[position])),
            ]
            for position in range(1, len(frame))
        ]
        axes[0].add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=config.gap_connector_linewidth,
                antialiaseds=True,
                zorder=1.5,
            )
        )
    figure.savefig(
        temporary_path,
        dpi=config.dpi,
        bbox_inches="tight",
        pad_inches=0,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(figure)

    with Image.open(temporary_path) as image:
        grayscale = image.convert("L")
        grayscale.save(temporary_path)
        width, height = grayscale.size
    temporary_path.replace(output_path)
    return {"imageWidth": width, "imageHeight": height, "imageMode": "L"}


def _calendar_anchors(
    start_date: str,
    end_date: str,
    every_months: int,
) -> list[pd.Timestamp]:
    start = pd.Timestamp(start_date).normalize()
    end = _effective_end_date(end_date)
    anchors: list[pd.Timestamp] = []
    step = 1
    anchor = (
        start
        + pd.DateOffset(months=step * every_months)
        - pd.Timedelta(days=1)
    )
    while anchor <= end:
        anchors.append(anchor.normalize())
        step += 1
        anchor = (
            start
            + pd.DateOffset(months=step * every_months)
            - pd.Timedelta(days=1)
        )
    return anchors


def _latest_required_end_date(config: PipelineConfig) -> pd.Timestamp:
    """현재 시점에서 완성된 6M/12M 기간 중 가장 늦은 종료일입니다."""
    anchors = [
        anchor
        for months in config.period_months
        for anchor in _calendar_anchors(config.start_date, config.end_date, months)
    ]
    if not anchors:
        raise ValueError("현재 날짜까지 완성된 6M/12M 구간이 없습니다.")
    return max(anchors)


def aggregate_ohlc_to_count(
    source: pd.DataFrame,
    target_candles: int,
) -> pd.DataFrame:
    """Aggregate every observed daily row into exactly N consecutive OHLC bins."""
    if len(source) < target_candles:
        raise ValueError(
            f"집계에는 최소 {target_candles}개 거래일이 필요합니다: {len(source)}개"
        )

    group_ids = np.floor(
        np.arange(len(source), dtype=float) * target_candles / len(source)
    ).astype(int)
    group_ids = np.clip(group_ids, 0, target_candles - 1)
    grouped = source.assign(_group=group_ids).groupby("_group", sort=True)
    aggregated = grouped.agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    end_positions = np.flatnonzero(
        np.r_[group_ids[1:] != group_ids[:-1], True]
    )
    aggregated.index = pd.DatetimeIndex(source.index[end_positions])
    aggregated.index.name = "Date"

    if len(aggregated) != target_candles:
        raise AssertionError(
            f"집계 결과가 {target_candles}개가 아닙니다: {len(aggregated)}개"
        )
    return aggregated


def build_period_windows(
    history: pd.DataFrame,
    period_months: int,
    config: PipelineConfig,
) -> list[dict]:
    """Build exact non-overlapping calendar periods from observed daily rows."""
    periods: list[dict] = []
    period_start = pd.Timestamp(config.start_date).normalize()
    for period_end in _calendar_anchors(
        config.start_date, config.end_date, period_months
    ):
        source = history.loc[
            (history.index >= period_start) & (history.index <= period_end)
        ].copy()
        candles: pd.DataFrame | None = None
        reason = ""
        if len(source) >= config.target_candles:
            candles = aggregate_ohlc_to_count(source, config.target_candles)
        else:
            reason = (
                f"기간 내 실제 거래일이 {len(source)}개라서 "
                f"{config.target_candles}캔들을 만들 수 없습니다."
            )
        periods.append(
            {
                "periodStart": period_start,
                "periodEnd": period_end,
                "source": source,
                "candles": candles,
                "reason": reason,
            }
        )
        period_start = period_end + pd.Timedelta(days=1)
    return periods


def load_cached_history(path: Path) -> pd.DataFrame:
    """저장된 토스 일봉 CSV를 다시 읽어 표준 인덱스로 복원합니다."""
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    frame = frame.sort_index()
    return frame


def _config_fingerprint(config: PipelineConfig) -> str:
    """체크포인트가 같은 생성 설정인지 판별하는 짧은 해시입니다."""
    payload = json.dumps(asdict(config), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    """Colab 중단 순간 CSV가 반쪽만 남지 않도록 임시 파일 후 교체합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_index_csv(frame: pd.DataFrame, path: Path) -> None:
    """날짜 인덱스가 필요한 OHLC CSV를 중단에 안전하게 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=True, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    """JSON 역시 임시 파일을 거쳐 원자적으로 저장합니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _checkpoint_reusable(manifest_path: Path, coverage_path: Path) -> bool:
    """완료 체크포인트와 그 이미지/CSV가 실제로 모두 존재하는지 확인합니다."""
    if not manifest_path.exists() or not coverage_path.exists():
        return False
    try:
        rows = pd.read_csv(manifest_path, dtype={"symbol": str})
        if rows.empty or rows["status"].eq("error").any():
            return False
        for row in rows.loc[rows["status"].eq("ok")].itertuples(index=False):
            image_path = Path(row.imagePath)
            candle_path = Path(row.candlePath)
            if not image_path.exists() or not candle_path.exists():
                return False
            # 중단 시 이름만 남은 손상 파일을 완료로 오인하지 않습니다.
            with Image.open(image_path) as image:
                image.verify()
            if len(pd.read_csv(candle_path)) != int(row.candles):
                return False
        return True
    except Exception:
        return False


def _collect_checkpoints(
    checkpoint_root: Path,
    universe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """여러 번 나눠 실행한 종목별 체크포인트를 하나의 표로 합칩니다."""
    allowed_symbols = set(universe["symbol"].map(_normalize_symbol))
    manifests: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []

    for path in sorted((checkpoint_root / "manifest").glob("*.csv")):
        if path.stem not in allowed_symbols:
            continue
        frame = pd.read_csv(path, dtype={"symbol": str})
        if not frame.empty:
            manifests.append(frame)

    for path in sorted((checkpoint_root / "coverage").glob("*.json")):
        if path.stem not in allowed_symbols:
            continue
        coverage_rows.append(json.loads(path.read_text(encoding="utf-8")))

    manifest = pd.concat(manifests, ignore_index=True) if manifests else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["rank", "frequency", "periodStart"], na_position="last"
        ).reset_index(drop=True)
    if not coverage.empty:
        coverage = coverage.sort_values("rank").reset_index(drop=True)
    return coverage, manifest


def run_pipeline(
    client_id: str,
    client_secret: str,
    output_root: str | Path,
    config: PipelineConfig | None = None,
    *,
    cache_root: str | Path | None = None,
    rank_start: int = 1,
    rank_end: int | None = None,
    resume: bool = True,
    force_rebuild: bool = False,
    refresh_universe: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """상위 N개 수집 → 6M/12M 이미지 생성을 실행합니다.

    Parameters
    ----------
    cache_root:
        원본 일봉 공동 캐시 위치입니다. 500개 실행 후 300개로 바꿔도 재사용됩니다.
    rank_start, rank_end:
        큰 작업을 1~100, 101~200처럼 나눌 때 사용합니다(양끝 포함).
    resume:
        완료된 종목 체크포인트가 유효하면 건너뜁니다.
    force_rebuild:
        True이면 체크포인트를 무시하고 이미지까지 다시 생성합니다.
    refresh_universe:
        False이면 첫 실행에서 저장한 시가총액 순위 스냅샷을 재사용합니다. 분할 실행
        도중 순위가 바뀌는 일을 막습니다. 새 기준일의 순위가 필요할 때만 True로
        실행하거나 새로운 OUTPUT_ROOT를 사용하세요.
    """
    config = config or PipelineConfig()
    if config.top_n < 1:
        raise ValueError("top_n은 1 이상이어야 합니다.")
    if rank_start < 1:
        raise ValueError("rank_start는 1 이상이어야 합니다.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cache_root = Path(cache_root) if cache_root is not None else output_root / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    client = TossMarketClient(
        client_id,
        client_secret,
        max_retries=config.max_retries,
        pause_seconds=config.request_pause_seconds,
    )

    universe_path = output_root / f"kospi_current_top{config.top_n}.csv"
    if universe_path.exists() and not refresh_universe:
        universe = pd.read_csv(universe_path, dtype={"symbol": str})
        universe["symbol"] = universe["symbol"].map(_normalize_symbol)
        if len(universe) != config.top_n:
            raise ValueError(
                f"저장된 종목 스냅샷은 {len(universe)}개인데 TOP_N은 "
                f"{config.top_n}입니다. OUTPUT_ROOT를 확인하세요."
            )
        universe_source = "saved_snapshot"
    else:
        universe = select_current_kospi_top_n(client, config)
        _atomic_csv(universe, universe_path)
        universe_source = "refreshed_from_api"

    universe_payload = universe[["rank", "symbol"]].to_csv(index=False)
    universe_fingerprint = hashlib.sha256(
        universe_payload.encode("utf-8")
    ).hexdigest()[:12]

    settings = asdict(config)
    settings["selectionMethod"] = "current sharesOutstanding * current lastPrice"
    settings["universeSource"] = universe_source
    settings["universeFingerprint"] = universe_fingerprint
    settings["marketDataEndpointsOnly"] = sorted(MARKET_ENDPOINTS)
    settings["configFingerprint"] = _config_fingerprint(config)
    settings["cacheRoot"] = str(cache_root)

    fingerprint = settings["configFingerprint"]
    checkpoint_id = f"{fingerprint}_{universe_fingerprint}"
    settings["checkpointId"] = checkpoint_id
    _atomic_json(settings, output_root / "settings.json")
    checkpoint_root = output_root / "checkpoints" / checkpoint_id
    (checkpoint_root / "manifest").mkdir(parents=True, exist_ok=True)
    (checkpoint_root / "coverage").mkdir(parents=True, exist_ok=True)

    effective_rank_end = min(rank_end or config.top_n, config.top_n)
    if rank_start > effective_rank_end:
        raise ValueError("rank_start가 rank_end보다 큽니다.")
    selected = universe.loc[
        universe["rank"].between(rank_start, effective_rank_end)
    ].copy()
    print(
        f"처리 범위: KOSPI 시가총액 {rank_start}~{effective_rank_end}위 "
        f"({len(selected)}종목), 전체 설정 TOP_N={config.top_n}"
    )

    for item in selected.itertuples(index=False):
        symbol = _normalize_symbol(item.symbol)
        name = str(item.name)
        rank = int(item.rank)
        stock_manifest_path = checkpoint_root / "manifest" / f"{symbol}.csv"
        stock_coverage_path = checkpoint_root / "coverage" / f"{symbol}.json"

        if (
            resume
            and not force_rebuild
            and _checkpoint_reusable(stock_manifest_path, stock_coverage_path)
        ):
            print(
                f"[{rank:03d}/{config.top_n:03d}] {name}({symbol}) "
                "완료 체크포인트 재사용"
            )
            continue

        required_end = _latest_required_end_date(config)
        cache_end = required_end.strftime("%Y%m%d")
        cache_start = pd.Timestamp(config.start_date).strftime("%Y%m%d")
        raw_path = (
            cache_root
            / "raw_daily"
            / f"{symbol}_{cache_start}_{cache_end}_adj{int(config.adjusted)}.csv"
        )
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{rank:03d}/{config.top_n:03d}] {name}({symbol}) 일봉 처리 중...")

        stock_manifest_rows: list[dict] = []
        stock_coverage: dict

        try:
            history: pd.DataFrame | None = None
            if raw_path.exists():
                try:
                    history = load_cached_history(raw_path)
                    source = "cache"
                except Exception as cache_error:
                    print(
                        f"  - 캐시를 읽지 못해 다시 받습니다: "
                        f"{type(cache_error).__name__}"
                    )
            if history is None:
                history = fetch_daily_history(
                    client,
                    symbol,
                    config.start_date,
                    required_end.date().isoformat(),
                    adjusted=config.adjusted,
                    page_size=config.page_size,
                )
                _atomic_index_csv(history, raw_path)
                source = "download"

            stock_coverage = {
                "rank": rank,
                "symbol": symbol,
                "name": name,
                "rows": len(history),
                "firstDate": history.index.min().date().isoformat()
                if len(history)
                else "",
                "lastDate": history.index.max().date().isoformat()
                if len(history)
                else "",
                "source": source,
                "status": "ok" if len(history) else "empty",
                "error": "",
            }
            print(f"  - 거래일 {len(history):,}개, 생성 준비 완료")

            for months in config.period_months:
                frequency = f"{months}M"
                for period in build_period_windows(history, months, config):
                    period_start = period["periodStart"]
                    period_end = period["periodEnd"]
                    source_window = period["source"]
                    window = period["candles"]
                    if window is None:
                        stock_manifest_rows.append(
                            {
                                "rank": rank,
                                "symbol": symbol,
                                "name": name,
                                "frequency": frequency,
                                "periodStart": period_start.date().isoformat(),
                                "periodEnd": period_end.date().isoformat(),
                                "firstTradingDay": "",
                                "lastTradingDay": "",
                                "sourceBars": len(source_window),
                                "candles": 0,
                                "weekendRows": 0,
                                "priceGapsConnected": config.connect_price_gaps,
                                "imagePath": "",
                                "candlePath": "",
                                "status": "skipped",
                                "error": period["reason"],
                            }
                        )
                        continue

                    stem = (
                        f"{rank:03d}_{symbol}_"
                        f"{period_start:%Y%m%d}_{period_end:%Y%m%d}"
                    )
                    image_path = output_root / "images" / frequency / f"{stem}.png"
                    candle_path = output_root / "candles" / frequency / f"{stem}.csv"
                    candle_path.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_index_csv(window, candle_path)
                    image_info = render_100_candles(window, image_path, config)

                    stock_manifest_rows.append(
                        {
                            "rank": rank,
                            "symbol": symbol,
                            "name": name,
                            "frequency": frequency,
                            "periodStart": period_start.date().isoformat(),
                            "periodEnd": period_end.date().isoformat(),
                            "firstTradingDay": source_window.index[0].date().isoformat(),
                            "lastTradingDay": source_window.index[-1].date().isoformat(),
                            "sourceBars": len(source_window),
                            "candles": len(window),
                            "weekendRows": int((window.index.dayofweek >= 5).sum()),
                            "priceGapsConnected": config.connect_price_gaps,
                            "imagePath": str(image_path),
                            "candlePath": str(candle_path),
                            "status": "ok",
                            "error": "",
                            **image_info,
                        }
                    )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            stock_coverage = {
                "rank": rank,
                "symbol": symbol,
                "name": name,
                "rows": 0,
                "firstDate": "",
                "lastDate": "",
                "source": "",
                "status": "error",
                "error": error,
            }
            stock_manifest_rows.append(
                {
                    "rank": rank,
                    "symbol": symbol,
                    "name": name,
                    "frequency": "",
                    "periodStart": "",
                    "periodEnd": "",
                    "firstTradingDay": "",
                    "lastTradingDay": "",
                    "sourceBars": 0,
                    "candles": 0,
                    "weekendRows": 0,
                    "imagePath": "",
                    "candlePath": "",
                    "status": "error",
                    "error": error,
                }
            )

        # 종목 단위로 끝날 때마다 저장하므로 런타임이 끊겨도 이 지점까지 보존됩니다.
        _atomic_csv(pd.DataFrame(stock_manifest_rows), stock_manifest_path)
        _atomic_json(stock_coverage, stock_coverage_path)

    # 현재까지 완료된 모든 분할 실행 결과를 최종 표로 다시 합칩니다.
    coverage, manifest = _collect_checkpoints(checkpoint_root, universe)
    _atomic_csv(coverage, output_root / "coverage.csv")
    _atomic_csv(manifest, output_root / "manifest.csv")
    return universe, coverage, manifest


def validate_manifest(manifest: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    successful = manifest.loc[manifest["status"].eq("ok")].copy()
    if successful.empty:
        return pd.DataFrame(
            [{"check": "successful images", "passed": False, "detail": "0"}]
        )

    period_start = pd.to_datetime(successful["periodStart"])
    period_end = pd.to_datetime(successful["periodEnd"])
    expected_end = pd.Series(
        [
            start + pd.DateOffset(months=int(frequency[:-1])) - pd.Timedelta(days=1)
            for start, frequency in zip(period_start, successful["frequency"])
        ],
        index=successful.index,
    )
    exact_periods = period_end.eq(pd.to_datetime(expected_end))

    checks = [
        {
            "check": "all images use exactly 100 candles",
            "passed": bool(successful["candles"].eq(config.target_candles).all()),
            "detail": str(successful["candles"].value_counts().to_dict()),
        },
        {
            "check": "no weekend rows were inserted",
            "passed": bool(successful["weekendRows"].eq(0).all()),
            "detail": str(int(successful["weekendRows"].sum())),
        },
        {
            "check": "only requested frequencies",
            "passed": set(successful["frequency"]).issubset(
                {f"{month}M" for month in config.period_months}
            ),
            "detail": str(sorted(successful["frequency"].unique())),
        },
        {
            "check": "every image covers its exact calendar period",
            "passed": bool(exact_periods.all()),
            "detail": f"{int(exact_periods.sum())}/{len(exact_periods)}",
        },
        {
            "check": "all top-N symbols produced images",
            "passed": int(successful["symbol"].nunique()) == config.top_n,
            "detail": f"{int(successful['symbol'].nunique())}/{config.top_n}",
        },
        {
            "check": "no per-symbol processing errors",
            "passed": not manifest["status"].eq("error").any(),
            "detail": str(int(manifest["status"].eq("error").sum())),
        },
    ]
    return pd.DataFrame(checks)


@dataclass(frozen=True)
class InferenceConfig:
    """학습 당시 DataLoader 전처리와 반드시 같아야 하는 CNN 추론 설정입니다."""

    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 64
    input_channels: int | None = None  # None이면 첫 Conv2d에서 1/3채널 자동 감지
    mean: tuple[float, ...] | None = None  # 학습 때 Normalize를 안 썼다면 None
    std: tuple[float, ...] | None = None
    top_k: int = 3
    # Colab/노트북에서는 0이 가장 안전합니다. 데이터가 매우 많고 문제가 없을 때만
    # 2~4로 올리세요.
    num_workers: int = 0
    device: str = "auto"


def _resolve_torch_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def load_pytorch_model(
    model_path: str | Path,
    *,
    model_format: str = "torchscript",
    model_builder: Callable[[], object] | None = None,
    device: str = "auto",
):
    """PyTorch 모델을 명시적인 저장 형식으로 안전하게 불러옵니다.

    ``model_format`` 선택값
    -----------------------
    torchscript:
        ``torch.jit.save``로 저장한 파일. 모델 클래스 코드 없이 바로 로드됩니다.
    state_dict:
        ``model.state_dict()`` 체크포인트. 학습 때와 같은 모델을 반환하는
        ``model_builder``가 반드시 필요합니다.
    full_model:
        ``torch.save(model, ...)`` 파일. pickle을 사용하므로 본인이 만든 신뢰할 수
        있는 파일에만 사용하세요.
    """
    import torch

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")

    resolved_device = _resolve_torch_device(torch, device)
    normalized_format = model_format.strip().lower()

    if normalized_format == "torchscript":
        model = torch.jit.load(str(model_path), map_location=resolved_device)
    elif normalized_format == "full_model":
        # PyTorch 2.6+에서는 full model을 위해 weights_only=False가 필요합니다.
        try:
            model = torch.load(
                str(model_path), map_location=resolved_device, weights_only=False
            )
        except TypeError:  # 구버전 PyTorch 호환
            model = torch.load(str(model_path), map_location=resolved_device)
        if not isinstance(model, torch.nn.Module):
            raise TypeError("full_model 파일에서 nn.Module을 찾지 못했습니다.")
    elif normalized_format == "state_dict":
        if model_builder is None:
            raise ValueError(
                "state_dict 형식은 학습 때와 동일한 모델을 만드는 "
                "model_builder 함수가 필요합니다."
            )
        model = model_builder()
        try:
            checkpoint = torch.load(
                str(model_path), map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(str(model_path), map_location="cpu")

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), dict):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("체크포인트에서 state_dict를 찾지 못했습니다.")

        # DataParallel로 학습한 파일의 'module.' 접두사를 자동 제거합니다.
        cleaned_state_dict = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state_dict.items()
        }
        model.load_state_dict(cleaned_state_dict, strict=True)
    else:
        raise ValueError(
            "model_format은 'torchscript', 'state_dict', 'full_model' 중 하나입니다."
        )

    model = model.to(resolved_device)
    model.eval()
    return model, resolved_device


def _infer_model_input_channels(model) -> int | None:
    """모델의 첫 Conv2d에서 입력 채널 수를 찾아냅니다."""
    try:
        import torch

        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                return int(module.in_channels)
            # TorchScript로 저장하면 Conv2d가 RecursiveScriptModule로 보일 수 있습니다.
            if getattr(module, "original_name", "") == "Conv2d":
                weight = getattr(module, "weight", None)
                if weight is not None and getattr(weight, "ndim", 0) == 4:
                    return int(weight.shape[1])
    except Exception:
        pass
    return None


def _normalize_class_names(
    class_names: Sequence[str] | dict | str | Path | None,
    num_classes: int,
) -> list[str]:
    """list, class_to_idx dict, JSON 경로를 출력 인덱스 순서의 이름으로 바꿉니다."""
    if class_names is None:
        return [f"class_{index}" for index in range(num_classes)]

    if isinstance(class_names, (str, Path)):
        payload = json.loads(Path(class_names).read_text(encoding="utf-8"))
    else:
        payload = class_names

    if isinstance(payload, dict):
        if all(isinstance(value, int) for value in payload.values()):
            # ImageFolder의 class_to_idx 형태: {"class_1": 0, ...}
            names = [""] * num_classes
            for name, index in payload.items():
                if 0 <= int(index) < num_classes:
                    names[int(index)] = str(name)
        elif all(str(key).isdigit() for key in payload):
            # {"0": "class_a"}와 {0: "class_a"}를 모두 허용합니다.
            names = [
                str(payload.get(str(index), payload.get(index, "")))
                for index in range(num_classes)
            ]
        else:
            raise ValueError("클래스 dict는 class_to_idx 또는 숫자 키 형식이어야 합니다.")
    else:
        names = [str(value) for value in payload]

    if len(names) != num_classes or any(not name for name in names):
        raise ValueError(
            f"클래스 이름은 모델 출력 {num_classes}개와 정확히 일치해야 합니다: "
            f"현재 {len(names)}개"
        )
    return names


def _extract_logits(model_output):
    """일반 Tensor, tuple, {'logits': Tensor} 모델 출력을 모두 지원합니다."""
    if isinstance(model_output, dict):
        if "logits" not in model_output:
            raise ValueError("모델 dict 출력에 'logits' 키가 없습니다.")
        return model_output["logits"]
    if isinstance(model_output, (tuple, list)):
        if not model_output:
            raise ValueError("모델 출력 tuple/list가 비어 있습니다.")
        return model_output[0]
    return model_output


def run_pytorch_inference(
    model,
    manifest: pd.DataFrame | str | Path,
    output_root: str | Path,
    *,
    config: InferenceConfig | None = None,
    class_names: Sequence[str] | dict | str | Path | None = None,
    frequencies: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """생성된 모든 정상 이미지를 CNN에 넣고 웹용 결과까지 저장합니다.

    반환값은 ``전체 예측``, ``종목·주기별 최신 예측``, ``클래스 요약``입니다.
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    inference_config = config or InferenceConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if isinstance(manifest, (str, Path)):
        manifest_frame = pd.read_csv(manifest, dtype={"symbol": str})
    else:
        manifest_frame = manifest.copy()

    rows = manifest_frame.loc[manifest_frame["status"].eq("ok")].copy()
    if frequencies is not None:
        rows = rows.loc[rows["frequency"].isin(list(frequencies))].copy()
    rows = rows.loc[rows["imagePath"].map(lambda value: Path(value).exists())]
    rows = rows.sort_values(["rank", "frequency", "periodStart"]).reset_index(drop=True)
    if rows.empty:
        raise ValueError("추론할 정상 이미지가 없습니다. manifest와 imagePath를 확인하세요.")

    resolved_device = _resolve_torch_device(torch, inference_config.device)
    model = model.to(resolved_device)
    model.eval()
    detected_channels = _infer_model_input_channels(model)
    input_channels = inference_config.input_channels or detected_channels or 1
    if input_channels not in (1, 3):
        raise ValueError(f"현재 파이프라인은 입력 채널 1 또는 3만 지원합니다: {input_channels}")

    mean = inference_config.mean
    std = inference_config.std
    if (mean is None) != (std is None):
        raise ValueError("mean과 std는 둘 다 지정하거나 둘 다 None이어야 합니다.")
    if mean is not None:
        if len(mean) == 1 and input_channels == 3:
            mean = tuple(mean) * 3
            std = tuple(std) * 3
        if len(mean) != input_channels or len(std) != input_channels:
            raise ValueError("mean/std 길이가 모델 입력 채널 수와 다릅니다.")

    class ChartDataset(Dataset):
        """manifest의 PNG를 학습 때와 같은 크기·채널·정규화로 변환합니다."""

        def __len__(self):
            return len(rows)

        def __getitem__(self, index: int):
            image_path = Path(rows.iloc[index]["imagePath"])
            with Image.open(image_path) as image:
                # 원본은 흑백입니다. 3채널 모델이면 같은 회색값을 RGB로 복제합니다.
                image = image.convert("L")
                target_height, target_width = inference_config.image_size
                image = image.resize(
                    (target_width, target_height),
                    resample=Image.Resampling.BILINEAR,
                )
                array = np.asarray(image, dtype=np.float32) / 255.0

            if input_channels == 1:
                array = array[None, :, :]
            else:
                array = np.repeat(array[None, :, :], 3, axis=0)
            tensor = torch.from_numpy(array)

            if mean is not None and std is not None:
                mean_tensor = torch.tensor(mean, dtype=tensor.dtype)[:, None, None]
                std_tensor = torch.tensor(std, dtype=tensor.dtype)[:, None, None]
                tensor = (tensor - mean_tensor) / std_tensor
            return tensor, index

    loader = DataLoader(
        ChartDataset(),
        batch_size=inference_config.batch_size,
        shuffle=False,
        num_workers=inference_config.num_workers,
        pin_memory=resolved_device.startswith("cuda"),
    )

    prediction_rows: list[dict] = []
    resolved_class_names: list[str] | None = None
    with torch.inference_mode():
        for batch_images, batch_indices in loader:
            logits = _extract_logits(model(batch_images.to(resolved_device)))
            if logits.ndim != 2:
                raise ValueError(f"모델 logits는 [batch, classes]여야 합니다: {logits.shape}")
            probabilities = torch.softmax(logits.float(), dim=1)
            if resolved_class_names is None:
                resolved_class_names = _normalize_class_names(
                    class_names, int(probabilities.shape[1])
                )
            top_k = min(inference_config.top_k, int(probabilities.shape[1]))
            top_values, top_indices = probabilities.topk(top_k, dim=1)

            for local_index, source_index in enumerate(batch_indices.tolist()):
                source = rows.iloc[int(source_index)].to_dict()
                # 웹 서버에서 OUTPUT_ROOT를 정적 파일 루트로 사용하기 쉽게 상대 경로도
                # 함께 기록합니다. 절대 경로 변환이 불가능한 경우 원래 값을 유지합니다.
                dataset_root = output_root.parent
                for path_key, relative_key in (
                    ("imagePath", "imageRelativePath"),
                    ("candlePath", "candleRelativePath"),
                ):
                    try:
                        source[relative_key] = Path(source[path_key]).relative_to(
                            dataset_root
                        ).as_posix()
                    except (KeyError, TypeError, ValueError):
                        source[relative_key] = str(source.get(path_key, ""))
                ranked = []
                for order in range(top_k):
                    class_index = int(top_indices[local_index, order].item())
                    ranked.append(
                        {
                            "rank": order + 1,
                            "index": class_index,
                            "label": resolved_class_names[class_index],
                            "confidence": float(top_values[local_index, order].item()),
                        }
                    )
                source.update(
                    {
                        "predictedIndex": ranked[0]["index"],
                        "predictedLabel": ranked[0]["label"],
                        "confidence": ranked[0]["confidence"],
                        "topK": json.dumps(ranked, ensure_ascii=False),
                    }
                )
                prediction_rows.append(source)

    predictions = pd.DataFrame(prediction_rows)
    predictions = predictions.sort_values(
        ["rank", "frequency", "periodStart"]
    ).reset_index(drop=True)
    latest = (
        predictions.sort_values("periodEnd")
        .groupby(["symbol", "frequency"], as_index=False)
        .tail(1)
        .sort_values(["rank", "frequency"])
        .reset_index(drop=True)
    )
    summary = (
        predictions.groupby(["frequency", "predictedIndex", "predictedLabel"])
        .agg(imageCount=("symbol", "size"), meanConfidence=("confidence", "mean"))
        .reset_index()
        .sort_values(["frequency", "imageCount"], ascending=[True, False])
    )

    _atomic_csv(predictions, output_root / "predictions_all.csv")
    _atomic_csv(latest, output_root / "predictions_latest.csv")
    _atomic_csv(summary, output_root / "prediction_summary.csv")

    # 이후 웹 백엔드가 바로 읽을 수 있는 JSON입니다.
    web_records = json.loads(
        latest.to_json(orient="records", force_ascii=False, date_format="iso")
    )
    for record in web_records:
        # CSV에서는 한 셀에 보관하기 위해 문자열이지만, 웹 JSON에서는 배열로 제공합니다.
        if isinstance(record.get("topK"), str):
            record["topK"] = json.loads(record["topK"])
    _atomic_json(
        {
            "generatedAt": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "recordCount": len(web_records),
            "records": web_records,
        },
        output_root / "web_results_latest.json",
    )
    _atomic_json(
        {
            "imageSize": list(inference_config.image_size),
            "inputChannels": input_channels,
            "mean": list(mean) if mean is not None else None,
            "std": list(std) if std is not None else None,
            "device": resolved_device,
            "classNames": resolved_class_names,
        },
        output_root / "inference_settings.json",
    )
    return predictions, latest, summary
