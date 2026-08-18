"""KOSPI OHLC 데이터를 6M/12M 100캔들 흑백 이미지로 변환하는 FastAPI.

이 서비스는 토스증권에 직접 접속하지 않습니다. 호출하는 백엔드가 토스증권에서
실제 일봉 OHLC를 조회한 뒤 ``POST /v1/render`` 요청의 ``candles`` 배열로 전달해야
합니다. 따라서 이 서비스에는 TOSS_CLIENT_ID/TOSS_CLIENT_SECRET이나 고정 outbound
IP가 필요하지 않습니다.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import matplotlib

# Render처럼 화면이 없는 서버에서도 matplotlib가 파일만 생성하도록 강제합니다.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from matplotlib.collections import LineCollection
from PIL import Image
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


LOGGER = logging.getLogger("kospi_chart_render_api")
KST = ZoneInfo("Asia/Seoul")
SYMBOL_PATTERN = re.compile(r"^\d{6}$")
REQUEST_ID_PATTERN = re.compile(r"^[0-9a-f]{20}$")
ALLOWED_PERIODS = frozenset({6, 12})
OHLC_COLUMNS = ["Open", "High", "Low", "Close"]
TARGET_CANDLES = 100
CHART_CACHE_VERSION = "ohlc-render-v2"


def _env_origins() -> tuple[str, ...]:
    value = os.getenv("CORS_ORIGINS", "")
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class APISettings:
    """Render 환경변수에서 읽는 비밀값과 실행 설정입니다."""

    storage_root: Path = Path("./render_storage")
    app_api_key: str | None = None
    cors_origins: tuple[str, ...] = ()
    public_base_url: str | None = None
    minimum_market_date: date = date(2016, 1, 1)
    target_candles: int = TARGET_CANDLES
    figsize_inches: tuple[float, float] = (3.0, 3.0)
    dpi: int = 100
    gap_connector_linewidth: float = 0.55

    @classmethod
    def from_env(cls) -> "APISettings":
        public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip() or None
        app_api_key = os.getenv("APP_API_KEY", "").strip() or None
        return cls(
            storage_root=Path(os.getenv("STORAGE_ROOT", "./render_storage")),
            app_api_key=app_api_key,
            cors_origins=_env_origins(),
            public_base_url=public_base_url,
            minimum_market_date=date.fromisoformat(
                os.getenv("MINIMUM_MARKET_DATE", "2016-01-01")
            ),
        )


class CandleInput(BaseModel):
    """일봉 한 개입니다.

    표준 필드(open/high/low/close)와 토스 응답 필드
    (openPrice/highPrice/lowPrice/closePrice)를 모두 받습니다. 토스의 volume,
    currency 같은 추가 필드는 이미지에 사용하지 않고 무시합니다.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    timestamp: str = Field(
        validation_alias=AliasChoices("timestamp", "date", "Date"),
        examples=["2023-08-14T00:00:00.000+09:00"],
    )
    open: float = Field(
        validation_alias=AliasChoices("open", "Open", "openPrice")
    )
    high: float = Field(
        validation_alias=AliasChoices("high", "High", "highPrice")
    )
    low: float = Field(
        validation_alias=AliasChoices("low", "Low", "lowPrice")
    )
    close: float = Field(
        validation_alias=AliasChoices("close", "Close", "closePrice")
    )

    @field_validator("timestamp", mode="before")
    @classmethod
    def validate_timestamp(cls, value: object) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("timestamp가 비어 있습니다.")
        try:
            parsed = pd.Timestamp(normalized)
        except Exception as error:
            raise ValueError(f"timestamp를 해석할 수 없습니다: {normalized}") from error
        if pd.isna(parsed):
            raise ValueError(f"timestamp를 해석할 수 없습니다: {normalized}")
        return normalized

    @model_validator(mode="after")
    def validate_prices(self) -> "CandleInput":
        prices = (self.open, self.high, self.low, self.close)
        if not all(math.isfinite(value) and value > 0 for value in prices):
            raise ValueError("OHLC 가격은 모두 0보다 큰 유한 숫자여야 합니다.")
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("high는 open/low/close보다 작을 수 없습니다.")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("low는 open/high/close보다 클 수 없습니다.")
        return self


class RenderRequest(BaseModel):
    """백엔드 서버가 토스 일봉을 조회한 뒤 보내는 이미지 변환 요청입니다."""

    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(description="6자리 국내 종목코드", examples=["005930"])
    name: str | None = Field(default=None, description="선택 입력 종목명")
    market: Literal["KOSPI"] = "KOSPI"
    as_of: date = Field(
        alias="asOf",
        description="6M/12M 구간의 기준일. 휴장일이면 직전 실제 거래일까지 사용",
        examples=["2023-08-14"],
    )
    periods: list[Literal[6, 12]] = Field(default_factory=lambda: [6, 12])
    adjusted: bool = Field(
        default=True,
        description="백엔드가 수정주가 일봉을 전달했는지 표시하는 메타데이터",
    )
    candles: list[CandleInput] = Field(
        min_length=1,
        max_length=5000,
        description=(
            "최대 요청 기간(보통 12개월)의 실제 일봉. 날짜순이 아니어도 서버가 정렬하며 "
            "주말/휴장일 가짜 행을 넣지 않습니다."
        ),
    )

    @field_validator("symbol", mode="before")
    @classmethod
    def validate_symbol(cls, value: object) -> str:
        normalized = str(value).strip()
        if not normalized.isdigit() or not 1 <= len(normalized) <= 6:
            raise ValueError("symbol은 1~6자리 숫자 종목코드여야 합니다.")
        return normalized.zfill(6)

    @field_validator("periods")
    @classmethod
    def validate_periods(cls, values: list[int]) -> list[int]:
        unique_values = list(dict.fromkeys(values))
        if not unique_values:
            raise ValueError("periods에는 6 또는 12가 하나 이상 있어야 합니다.")
        if any(value not in ALLOWED_PERIODS for value in unique_values):
            raise ValueError("periods는 6과 12만 지원합니다.")
        return unique_values


class PeriodResult(BaseModel):
    period: Literal["6M", "12M"]
    period_start: date = Field(alias="periodStart")
    period_end: date = Field(alias="periodEnd")
    first_trading_day: date | None = Field(alias="firstTradingDay")
    last_trading_day: date | None = Field(alias="lastTradingDay")
    source_bars: int = Field(alias="sourceBars")
    candles: int
    status: Literal["ok", "insufficient_data"]
    message: str | None = None
    image_url: str | None = Field(alias="imageUrl", default=None)
    candle_url: str | None = Field(alias="candleUrl", default=None)
    image_width: int | None = Field(alias="imageWidth", default=None)
    image_height: int | None = Field(alias="imageHeight", default=None)
    image_mode: str | None = Field(alias="imageMode", default=None)

    model_config = ConfigDict(populate_by_name=True)


class RenderResponse(BaseModel):
    request_id: str = Field(alias="requestId")
    symbol: str
    name: str
    market: Literal["KOSPI"]
    as_of: date = Field(alias="asOf")
    requested_periods: list[str] = Field(alias="requestedPeriods")
    target_candles: int = Field(alias="targetCandles")
    input_bars: int = Field(alias="inputBars")
    usable_bars: int = Field(alias="usableBars")
    duplicates_removed: int = Field(alias="duplicatesRemoved")
    adjusted: bool
    cached: bool
    results: list[PeriodResult]

    model_config = ConfigDict(populate_by_name=True)


def rolling_window_start(as_of: date, months: int) -> date:
    """종료일을 포함하는 정확한 N개월 구간의 시작일을 구합니다.

    예: 2023-08-14 종료 6M은 2023-02-15부터입니다.
    """

    timestamp = pd.Timestamp(as_of)
    return (
        timestamp + pd.Timedelta(days=1) - pd.DateOffset(months=months)
    ).date()


def _timestamp_to_kst_day(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(KST).tz_localize(None)
    return timestamp.normalize()


def normalize_history(candles: list[CandleInput]) -> tuple[pd.DataFrame, int]:
    """입력 일봉을 KST 날짜 기준으로 정렬하고 중복 날짜를 제거합니다."""

    rows = [
        {
            "Date": _timestamp_to_kst_day(item.timestamp),
            "Open": float(item.open),
            "High": float(item.high),
            "Low": float(item.low),
            "Close": float(item.close),
        }
        for item in candles
    ]
    frame = pd.DataFrame(rows)
    duplicate_count = int(frame.duplicated("Date", keep="last").sum())
    frame = frame.drop_duplicates("Date", keep="last").sort_values("Date")
    frame = frame.set_index("Date")
    frame.index = pd.DatetimeIndex(frame.index)
    frame.index.name = "Date"

    weekend_dates = frame.index[frame.index.dayofweek >= 5]
    if len(weekend_dates):
        preview = ", ".join(item.date().isoformat() for item in weekend_dates[:3])
        raise ValueError(
            "주말 캔들은 입력할 수 없습니다. 실제 거래일만 보내세요: " + preview
        )
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise AssertionError("정규화 후 날짜 인덱스가 올바르지 않습니다.")
    return frame[OHLC_COLUMNS], duplicate_count


def aggregate_ohlc_to_count(
    source: pd.DataFrame,
    target_candles: int = TARGET_CANDLES,
) -> pd.DataFrame:
    """관측된 실제 일봉을 순서대로 묶어 정확히 N개의 OHLC로 집계합니다."""

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
    end_positions = np.flatnonzero(np.r_[group_ids[1:] != group_ids[:-1], True])
    aggregated.index = pd.DatetimeIndex(source.index[end_positions])
    aggregated.index.name = "Date"

    if len(aggregated) != target_candles:
        raise AssertionError(
            f"집계 결과가 {target_candles}개가 아닙니다: {len(aggregated)}개"
        )
    return aggregated


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
    output_path: Path,
    settings: APISettings,
) -> dict:
    """학습 데이터와 같은 흑백·무축·무격자 PNG를 생성합니다."""

    if len(frame) != settings.target_candles:
        raise ValueError(
            f"이미지 입력은 {settings.target_candles}개여야 합니다: {len(frame)}개"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.name + ".tmp.png")

    figure, axes = mpf.plot(
        frame[OHLC_COLUMNS],
        type="candle",
        style=_monochrome_style(),
        volume=False,
        axisoff=True,
        show_nontrading=False,
        figsize=settings.figsize_inches,
        tight_layout=False,
        returnfig=True,
        warn_too_much_data=1000,
    )
    for axis in axes:
        axis.set_axis_off()
        axis.grid(False)

    # 연속된 캔들의 전 종가와 다음 시가를 연결해 이미지 중간의 시각적 빈 구간을
    # 방지합니다. 주말/휴장일 행을 새로 만들지는 않습니다.
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
            linewidths=settings.gap_connector_linewidth,
            antialiaseds=True,
            zorder=1.5,
        )
    )
    figure.savefig(
        temporary_path,
        dpi=settings.dpi,
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


def _atomic_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_index_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=True, encoding="utf-8-sig")
    temporary.replace(path)


class ChartRenderService:
    """입력 검증, 100캔들 집계, 이미지 생성과 캐시를 담당합니다."""

    def __init__(self, settings: APISettings) -> None:
        self.settings = settings
        self.settings.storage_root.mkdir(parents=True, exist_ok=True)
        self._operation_lock = threading.RLock()

    def _request_root(self, request_id: str) -> Path:
        return self.settings.storage_root / "requests" / request_id

    def _period_paths(self, request_id: str, months: int) -> dict[str, Path]:
        root = self._request_root(request_id) / f"{months}M"
        return {
            "image": root / "chart.png",
            "candles": root / "candles.csv",
            "metadata": root / "metadata.json",
        }

    @staticmethod
    def _load_period_cache(paths: dict[str, Path]) -> dict | None:
        if not paths["metadata"].exists():
            return None
        try:
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if metadata.get("status") == "insufficient_data":
                return metadata
            if metadata.get("status") != "ok" or metadata.get("candles") != 100:
                return None
            if not paths["image"].exists() or not paths["candles"].exists():
                return None
            with Image.open(paths["image"]) as image:
                image.verify()
            if len(pd.read_csv(paths["candles"])) != 100:
                return None
            return metadata
        except Exception:
            return None

    def _make_request_id(
        self,
        payload: RenderRequest,
        history: pd.DataFrame,
    ) -> str:
        history_text = history.to_csv(
            index=True,
            date_format="%Y-%m-%d",
            float_format="%.10g",
        )
        identity = {
            "version": CHART_CACHE_VERSION,
            "symbol": payload.symbol,
            "asOf": payload.as_of.isoformat(),
            "periods": list(payload.periods),
            "adjusted": payload.adjusted,
            "historySha256": hashlib.sha256(history_text.encode("utf-8")).hexdigest(),
        }
        encoded = json.dumps(identity, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:20]

    def _build_period(
        self,
        request_id: str,
        as_of: date,
        months: int,
        history: pd.DataFrame,
    ) -> tuple[dict, bool]:
        paths = self._period_paths(request_id, months)
        cached = self._load_period_cache(paths)
        if cached is not None:
            return cached, True

        period_start = rolling_window_start(as_of, months)
        source = history.loc[
            (history.index >= pd.Timestamp(period_start))
            & (history.index <= pd.Timestamp(as_of))
        ].copy()
        base = {
            "period": f"{months}M",
            "periodStart": period_start.isoformat(),
            "periodEnd": as_of.isoformat(),
            "sourceBars": len(source),
        }

        if len(source) < self.settings.target_candles:
            result = {
                **base,
                "firstTradingDay": (
                    source.index.min().date().isoformat() if len(source) else None
                ),
                "lastTradingDay": (
                    source.index.max().date().isoformat() if len(source) else None
                ),
                "candles": 0,
                "status": "insufficient_data",
                "message": (
                    f"전달받은 구간의 실제 일봉이 {len(source)}개라 "
                    f"{self.settings.target_candles}캔들을 만들 수 없습니다. "
                    "백엔드의 토스 페이지네이션과 조회 시작일을 확인하세요."
                ),
            }
            _atomic_json(result, paths["metadata"])
            return result, False

        candles = aggregate_ohlc_to_count(source, self.settings.target_candles)
        _atomic_index_csv(candles, paths["candles"])
        image_info = render_100_candles(candles, paths["image"], self.settings)
        result = {
            **base,
            "firstTradingDay": source.index.min().date().isoformat(),
            "lastTradingDay": source.index.max().date().isoformat(),
            "candles": len(candles),
            "status": "ok",
            "message": None,
            **image_info,
        }
        _atomic_json(result, paths["metadata"])
        return result, False

    def render(self, payload: RenderRequest) -> dict:
        if payload.as_of > datetime.now(tz=KST).date():
            raise ValueError(f"미래 기준일은 요청할 수 없습니다: {payload.as_of}")

        periods = [int(value) for value in payload.periods]
        earliest_start = min(
            rolling_window_start(payload.as_of, months) for months in periods
        )
        if earliest_start < self.settings.minimum_market_date:
            raise ValueError(
                f"{max(periods)}M 시작일 {earliest_start}이 지원 시작일 "
                f"{self.settings.minimum_market_date}보다 빠릅니다."
            )

        history, duplicates_removed = normalize_history(payload.candles)
        input_bars = len(payload.candles)
        usable_history = history.loc[
            (history.index >= pd.Timestamp(earliest_start))
            & (history.index <= pd.Timestamp(payload.as_of))
        ].copy()
        if usable_history.empty:
            raise ValueError("요청한 기간에 해당하는 일봉이 없습니다.")

        request_id = self._make_request_id(payload, usable_history)
        with self._operation_lock:
            results: list[dict] = []
            cache_flags: list[bool] = []
            for months in periods:
                result, cached = self._build_period(
                    request_id,
                    payload.as_of,
                    months,
                    usable_history,
                )
                results.append(result)
                cache_flags.append(cached)

        return {
            "requestId": request_id,
            "symbol": payload.symbol,
            "name": (payload.name or payload.symbol).strip() or payload.symbol,
            "market": "KOSPI",
            "asOf": payload.as_of.isoformat(),
            "requestedPeriods": [f"{months}M" for months in periods],
            "targetCandles": self.settings.target_candles,
            "inputBars": input_bars,
            "usableBars": len(usable_history),
            "duplicatesRemoved": duplicates_removed,
            "adjusted": payload.adjusted,
            "cached": all(cache_flags),
            "results": results,
        }

    def artifact_path(self, request_id: str, period: str, kind: str) -> Path:
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise ValueError("잘못된 requestId입니다.")
        normalized = period.upper()
        if normalized not in {"6M", "12M"}:
            raise ValueError("period는 6M 또는 12M이어야 합니다.")
        paths = self._period_paths(request_id, int(normalized[:-1]))
        return paths[kind]


def _file_url(
    request: Request,
    route_name: str,
    settings: APISettings,
    *,
    request_id: str,
    period: str,
) -> str:
    if not settings.public_base_url:
        return str(
            request.url_for(route_name, request_id=request_id, period=period)
        )
    path = request.app.url_path_for(
        route_name,
        request_id=request_id,
        period=period,
    )
    return settings.public_base_url.rstrip("/") + str(path)


def create_app(
    settings: APISettings | None = None,
    service: ChartRenderService | None = None,
) -> FastAPI:
    settings = settings or APISettings.from_env()
    app = FastAPI(
        title="KOSPI OHLC 100-Candle Render API",
        version="2.0.0",
        description=(
            "백엔드가 조회한 KOSPI 실제 일봉 OHLC를 받아 6M/12M 각각을 정확히 "
            "100캔들 흑백 이미지로 변환합니다. 이 API는 토스증권에 직접 접속하지 "
            "않으며 계좌·토스 Client ID·Client Secret을 사용하지 않습니다."
        ),
    )
    app.state.settings = settings
    app.state.service = service
    app.state.service_init_lock = threading.Lock()

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key"],
        )

    def get_service() -> ChartRenderService:
        if app.state.service is None:
            with app.state.service_init_lock:
                if app.state.service is None:
                    try:
                        app.state.service = ChartRenderService(settings)
                    except Exception as error:
                        LOGGER.exception("이미지 변환 서비스 초기화 실패")
                        raise HTTPException(
                            status_code=503,
                            detail="STORAGE_ROOT 경로와 쓰기 권한을 확인하세요.",
                        ) from error
        return app.state.service

    def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
        expected = settings.app_api_key
        if expected is None:
            return
        supplied = x_api_key or ""
        if not hmac.compare_digest(supplied, expected):
            raise HTTPException(status_code=401, detail="유효한 X-API-Key가 필요합니다.")

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "mode": "ohlc-render-only",
            "tossConnection": False,
            "tossCredentialsRequired": False,
            "targetCandles": settings.target_candles,
        }

    @app.post(
        "/v1/render",
        response_model=RenderResponse,
        response_model_by_alias=True,
        dependencies=[Depends(require_api_key)],
    )
    def render(
        payload: RenderRequest,
        web_request: Request,
        service_instance: ChartRenderService = Depends(get_service),
    ) -> dict:
        try:
            response = service_instance.render(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except OSError as error:
            LOGGER.exception("이미지 또는 CSV 저장 실패")
            raise HTTPException(
                status_code=503,
                detail="이미지 저장소를 사용할 수 없습니다.",
            ) from error
        except Exception as error:
            LOGGER.exception("이미지 변환 중 예기치 않은 오류")
            raise HTTPException(
                status_code=500,
                detail="서버 내부 이미지 변환 오류가 발생했습니다.",
            ) from error

        for result in response["results"]:
            if result["status"] != "ok":
                continue
            result["imageUrl"] = _file_url(
                web_request,
                "get_image",
                settings,
                request_id=response["requestId"],
                period=result["period"],
            )
            result["candleUrl"] = _file_url(
                web_request,
                "get_candles",
                settings,
                request_id=response["requestId"],
                period=result["period"],
            )
        return response

    @app.get(
        "/v1/images/{request_id}/{period}.png",
        name="get_image",
    )
    def get_image(
        request_id: str,
        period: str,
        service_instance: ChartRenderService = Depends(get_service),
    ) -> FileResponse:
        try:
            path = service_instance.artifact_path(request_id, period, "image")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not path.exists():
            raise HTTPException(status_code=404, detail="생성된 이미지가 없습니다.")
        return FileResponse(
            path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get(
        "/v1/candles/{request_id}/{period}.csv",
        name="get_candles",
    )
    def get_candles(
        request_id: str,
        period: str,
        service_instance: ChartRenderService = Depends(get_service),
    ) -> FileResponse:
        try:
            path = service_instance.artifact_path(request_id, period, "candles")
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not path.exists():
            raise HTTPException(status_code=404, detail="생성된 100캔들 CSV가 없습니다.")
        return FileResponse(
            path,
            media_type="text/csv; charset=utf-8",
            filename=f"{request_id}_{period.upper()}_100candles.csv",
        )

    return app


app = create_app()
