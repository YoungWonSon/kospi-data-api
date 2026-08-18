"""종목코드와 기준일을 받아 6M/12M 차트와 CNN 결과를 만드는 FastAPI 서버.

이 파일은 ``toss_kospi_production_pipeline.py``의 검증된 수집·100캔들 집계·
이미지·PyTorch 추론 함수를 그대로 재사용합니다. 토스 Client ID/Secret과 서비스용
API Key는 코드나 응답에 넣지 않고 환경변수에서만 읽습니다.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Literal
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

import toss_kospi_production_pipeline as pipeline


LOGGER = logging.getLogger("toss_kospi_ondemand_api")
KST = ZoneInfo("Asia/Seoul")
SYMBOL_PATTERN = re.compile(r"^\d{6}$")
ALLOWED_PERIODS = frozenset({6, 12})
CHART_CACHE_VERSION = "chart-v1"


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    return int(value) if value else None


def _env_float_tuple(name: str) -> tuple[float, ...] | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    return tuple(float(item.strip()) for item in value.split(","))


def _env_origins() -> tuple[str, ...]:
    value = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return tuple(origin.strip() for origin in value.split(",") if origin.strip())


@dataclass(frozen=True)
class APISettings:
    """서버 환경설정. 실제 비밀 값은 이 객체를 직렬화하거나 출력하지 않습니다."""

    toss_client_id: str
    toss_client_secret: str
    storage_root: Path = Path("./ondemand_storage")
    universe_csv: Path | None = None
    app_api_key: str | None = None
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    minimum_market_date: date = date(2016, 1, 1)
    adjusted: bool = True
    request_pause_seconds: float = 0.23
    max_retries: int = 6
    page_size: int = 200
    public_base_url: str | None = None

    # 모델 경로가 비어 있으면 predict=false 요청으로 이미지만 생성할 수 있습니다.
    model_path: Path | None = None
    model_format: str = "torchscript"
    model_builder_path: str | None = None
    class_names_path: Path | None = None
    model_image_size: tuple[int, int] = (224, 224)
    model_input_channels: int | None = None
    model_mean: tuple[float, ...] | None = None
    model_std: tuple[float, ...] | None = None
    model_top_k: int = 3
    model_device: str = "auto"

    @classmethod
    def from_env(cls) -> "APISettings":
        universe_value = os.getenv("UNIVERSE_CSV", "").strip()
        model_value = os.getenv("MODEL_PATH", "").strip()
        class_names_value = os.getenv("CLASS_NAMES_PATH", "").strip()
        public_base_url = os.getenv("PUBLIC_BASE_URL", "").strip() or None
        app_api_key = os.getenv("APP_API_KEY", "").strip() or None
        return cls(
            toss_client_id=os.getenv("TOSS_CLIENT_ID", "").strip(),
            toss_client_secret=os.getenv("TOSS_CLIENT_SECRET", "").strip(),
            storage_root=Path(os.getenv("STORAGE_ROOT", "./ondemand_storage")),
            universe_csv=Path(universe_value) if universe_value else None,
            app_api_key=app_api_key,
            cors_origins=_env_origins(),
            minimum_market_date=date.fromisoformat(
                os.getenv("MINIMUM_MARKET_DATE", "2016-01-01")
            ),
            adjusted=_env_bool("ADJUSTED", True),
            request_pause_seconds=float(os.getenv("REQUEST_PAUSE_SECONDS", "0.23")),
            max_retries=int(os.getenv("MAX_RETRIES", "6")),
            page_size=int(os.getenv("PAGE_SIZE", "200")),
            public_base_url=public_base_url,
            model_path=Path(model_value) if model_value else None,
            model_format=os.getenv("MODEL_FORMAT", "torchscript").strip(),
            model_builder_path=os.getenv("MODEL_BUILDER", "").strip() or None,
            class_names_path=(
                Path(class_names_value) if class_names_value else None
            ),
            model_image_size=(
                int(os.getenv("MODEL_IMAGE_HEIGHT", "224")),
                int(os.getenv("MODEL_IMAGE_WIDTH", "224")),
            ),
            model_input_channels=_env_optional_int("MODEL_INPUT_CHANNELS"),
            model_mean=_env_float_tuple("MODEL_MEAN"),
            model_std=_env_float_tuple("MODEL_STD"),
            model_top_k=int(os.getenv("MODEL_TOP_K", "3")),
            model_device=os.getenv("MODEL_DEVICE", "auto").strip(),
        )


class AnalyzeRequest(BaseModel):
    """프런트엔드가 보내는 분석 요청."""

    model_config = ConfigDict(populate_by_name=True)

    symbol: str = Field(
        description="국내 종목코드. 부족한 앞의 0은 서버가 보정",
        examples=["005930"],
    )
    as_of: date = Field(
        alias="asOf",
        description="이 날짜를 포함해 과거 6/12개월을 계산",
        examples=["2023-08-14"],
    )
    periods: list[Literal[6, 12]] = Field(default_factory=lambda: [6, 12])
    predict: bool = True

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
        return unique_values


class TopKPrediction(BaseModel):
    rank: int
    index: int
    label: str
    confidence: float


class PredictionResult(BaseModel):
    predicted_index: int = Field(alias="predictedIndex")
    predicted_label: str = Field(alias="predictedLabel")
    confidence: float
    top_k: list[TopKPrediction] = Field(alias="topK")

    model_config = ConfigDict(populate_by_name=True)


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
    prediction: PredictionResult | None = None

    model_config = ConfigDict(populate_by_name=True)


class AnalyzeResponse(BaseModel):
    request_id: str = Field(alias="requestId")
    symbol: str
    name: str
    market: str
    as_of: date = Field(alias="asOf")
    requested_periods: list[str] = Field(alias="requestedPeriods")
    target_candles: int = Field(alias="targetCandles")
    adjusted: bool
    cached: bool
    results: list[PeriodResult]

    model_config = ConfigDict(populate_by_name=True)


def rolling_window_start(as_of: date, months: int) -> date:
    """기준일을 포함하는 정확한 N개월 구간의 시작일을 계산합니다.

    기존 반기 구간의 ``시작 + N개월 - 1일 = 종료`` 규칙과 동일합니다.
    예: 2023-08-14 종료 6M -> 2023-02-15 시작.
    """
    timestamp = pd.Timestamp(as_of)
    return (
        timestamp + pd.Timedelta(days=1) - pd.DateOffset(months=months)
    ).date()


def _load_builder(import_path: str | None) -> Callable[[], object] | None:
    if not import_path:
        return None
    if ":" not in import_path:
        raise ValueError("MODEL_BUILDER는 'python_module:function_name' 형식이어야 합니다.")
    module_name, function_name = import_path.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"MODEL_BUILDER가 callable이 아닙니다: {import_path}")
    return function


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


def _read_ohlc(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="Date", parse_dates=["Date"])
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    return frame.sort_index()


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class OnDemandChartService:
    """토스 조회, 결과 캐시, 이미지 생성, CNN 추론을 한곳에서 조정합니다."""

    def __init__(
        self,
        settings: APISettings,
        *,
        client: pipeline.TossMarketClient | None = None,
        model=None,
    ) -> None:
        if not settings.toss_client_id or not settings.toss_client_secret:
            raise ValueError("TOSS_CLIENT_ID와 TOSS_CLIENT_SECRET 환경변수가 필요합니다.")
        if settings.model_input_channels not in (None, 1, 3):
            raise ValueError("MODEL_INPUT_CHANNELS는 빈 값, 1, 3 중 하나여야 합니다.")
        if (settings.model_mean is None) != (settings.model_std is None):
            raise ValueError("MODEL_MEAN과 MODEL_STD는 둘 다 지정하거나 둘 다 비워야 합니다.")
        if settings.class_names_path is not None and not settings.class_names_path.exists():
            raise FileNotFoundError(
                f"CLASS_NAMES_PATH를 찾을 수 없습니다: {settings.class_names_path}"
            )

        self.settings = settings
        self.settings.storage_root.mkdir(parents=True, exist_ok=True)
        self.client = client or pipeline.TossMarketClient(
            settings.toss_client_id,
            settings.toss_client_secret,
            max_retries=settings.max_retries,
            pause_seconds=settings.request_pause_seconds,
        )
        self._operation_lock = threading.RLock()
        self._stock_cache: dict[str, dict] = {}
        self._universe = self._load_universe(settings.universe_csv)

        self.inference_config = pipeline.InferenceConfig(
            image_size=settings.model_image_size,
            batch_size=2,
            input_channels=settings.model_input_channels,
            mean=settings.model_mean,
            std=settings.model_std,
            top_k=settings.model_top_k,
            num_workers=0,
            device=settings.model_device,
        )
        self.class_names = settings.class_names_path
        self.model = model
        self.model_device: str | None = None
        if self.model is None and settings.model_path is not None:
            self.model, self.model_device = pipeline.load_pytorch_model(
                settings.model_path,
                model_format=settings.model_format,
                model_builder=_load_builder(settings.model_builder_path),
                device=settings.model_device,
            )
        elif self.model is not None:
            # 테스트나 애플리케이션 코드가 이미 로드한 모델을 주입할 수도 있습니다.
            self.model_device = settings.model_device

        self.model_fingerprint = self._model_fingerprint()

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @staticmethod
    def _load_universe(path: Path | None) -> pd.DataFrame | None:
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"UNIVERSE_CSV를 찾을 수 없습니다: {path}")
        frame = pd.read_csv(path, dtype={"symbol": str})
        required = {"symbol", "name"}
        if not required.issubset(frame.columns):
            raise ValueError(f"UNIVERSE_CSV 필수 열 누락: {sorted(required - set(frame))}")
        frame["symbol"] = frame["symbol"].str.zfill(6)
        return frame.drop_duplicates("symbol").set_index("symbol", drop=False)

    def _model_fingerprint(self) -> str:
        if self.model is None:
            return "no-model"
        payload = {
            "modelPath": str(self.settings.model_path or "injected-model"),
            "modelFormat": self.settings.model_format,
            "inference": asdict(self.inference_config),
            "classNamesPath": str(self.settings.class_names_path or ""),
        }
        for path_key, path in (
            ("model", self.settings.model_path),
            ("classes", self.settings.class_names_path),
        ):
            if path is not None and path.exists():
                stat = path.stat()
                payload[f"{path_key}Size"] = stat.st_size
                payload[f"{path_key}MtimeNs"] = stat.st_mtime_ns
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:12]

    def _resolve_stock(self, symbol: str) -> dict:
        if symbol in self._stock_cache:
            return self._stock_cache[symbol]

        if self._universe is not None:
            if symbol not in self._universe.index:
                raise ValueError("현재 서비스 허용 종목 목록에 없는 종목코드입니다.")
            row = self._universe.loc[symbol]
            rank_value = row.get("rank", 0)
            stock = {
                "symbol": symbol,
                "name": str(row["name"]),
                "market": str(row.get("market", "KOSPI")),
                "rank": int(rank_value) if pd.notna(rank_value) else 0,
            }
            self._stock_cache[symbol] = stock
            return stock

        result = self.client.get(
            "/api/v1/stocks", params={"symbols": symbol}
        ).get("result", [])
        if not result:
            raise ValueError("토스증권에서 종목코드를 찾지 못했습니다.")
        item = result[0]
        valid = (
            str(item.get("market", "")).upper() == "KOSPI"
            and str(item.get("securityType", "")).upper() == "STOCK"
            and _truthy(item.get("isCommonShare", False))
            and str(item.get("status", "")).upper() == "ACTIVE"
        )
        if not valid:
            raise ValueError("현재 ACTIVE KOSPI 보통주만 요청할 수 있습니다.")
        stock = {
            "symbol": symbol,
            "name": str(item.get("name", symbol)),
            "market": "KOSPI",
            "rank": 0,
        }
        self._stock_cache[symbol] = stock
        return stock

    def _validate_request_date(self, as_of: date, periods: list[int]) -> None:
        today_kst = datetime.now(tz=KST).date()
        if as_of > today_kst:
            raise ValueError(f"미래 기준일은 요청할 수 없습니다: {as_of}")
        earliest_start = min(rolling_window_start(as_of, value) for value in periods)
        if earliest_start < self.settings.minimum_market_date:
            raise ValueError(
                f"{max(periods)}M 시작일 {earliest_start}이 지원 시작일 "
                f"{self.settings.minimum_market_date}보다 빠릅니다."
            )

    def _request_root(self, symbol: str, as_of: date) -> Path:
        chart_variant = f"{CHART_CACHE_VERSION}_adj{int(self.settings.adjusted)}"
        return (
            self.settings.storage_root
            / "requests"
            / chart_variant
            / symbol
            / as_of.isoformat()
        )

    def _period_paths(self, symbol: str, as_of: date, months: int) -> dict[str, Path]:
        root = self._request_root(symbol, as_of) / f"{months}M"
        return {
            "root": root,
            "image": root / "chart.png",
            "candles": root / "candles.csv",
            "metadata": root / "metadata.json",
        }

    @staticmethod
    def _load_period_cache(paths: dict[str, Path]) -> dict | None:
        """정상 이미지뿐 아니라 '거래일 부족' 결과도 같은 요청에서 재사용합니다."""
        if not paths["metadata"].exists():
            return None
        try:
            from PIL import Image

            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            if metadata.get("status") == "insufficient_data":
                return metadata
            if metadata.get("status") != "ok" or int(metadata.get("candles", 0)) != 100:
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

    def _load_or_fetch_history(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        raw_path = (
            self.settings.storage_root
            / "raw_requests"
            / symbol
            / f"{start.isoformat()}_{end.isoformat()}_adj{int(self.settings.adjusted)}.csv"
        )
        if raw_path.exists():
            try:
                return _read_ohlc(raw_path)
            except Exception:
                LOGGER.warning("손상된 원본 캐시를 다시 받습니다: %s", raw_path)

        history = pipeline.fetch_daily_history(
            self.client,
            symbol,
            start.isoformat(),
            end.isoformat(),
            adjusted=self.settings.adjusted,
            page_size=self.settings.page_size,
        )
        _atomic_index_csv(history, raw_path)
        return history

    def _build_period(
        self,
        stock: dict,
        as_of: date,
        months: int,
        history: pd.DataFrame,
    ) -> tuple[dict, bool]:
        paths = self._period_paths(stock["symbol"], as_of, months)
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
        if len(source) < 100:
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
                    f"실제 거래일이 {len(source)}개라 100캔들을 만들 수 없습니다. "
                    "신규 상장·장기 거래정지 여부를 확인하세요."
                ),
            }
            _atomic_json(result, paths["metadata"])
            return result, False

        candles = pipeline.aggregate_ohlc_to_count(source, 100)
        chart_config = pipeline.PipelineConfig(
            start_date=period_start.isoformat(),
            end_date=as_of.isoformat(),
            top_n=1,
            target_candles=100,
            period_months=(months,),
            adjusted=self.settings.adjusted,
            figsize_inches=(3, 3),
            dpi=100,
            connect_price_gaps=True,
            gap_connector_linewidth=0.55,
        )
        _atomic_index_csv(candles, paths["candles"])
        image_info = pipeline.render_100_candles(candles, paths["image"], chart_config)
        result = {
            **base,
            "firstTradingDay": source.index.min().date().isoformat(),
            "lastTradingDay": source.index.max().date().isoformat(),
            "candles": len(candles),
            "status": "ok",
            "message": None,
            "imagePath": str(paths["image"]),
            "candlePath": str(paths["candles"]),
            **image_info,
        }
        _atomic_json(result, paths["metadata"])
        return result, False

    def _run_inference(
        self,
        stock: dict,
        as_of: date,
        period_results: list[dict],
    ) -> dict[str, dict]:
        if self.model is None:
            raise RuntimeError(
                "CNN 모델이 설정되지 않았습니다. MODEL_PATH를 설정하거나 "
                "predict=false로 요청하세요."
            )

        valid = [result for result in period_results if result["status"] == "ok"]
        if not valid:
            return {}
        inference_root = (
            self._request_root(stock["symbol"], as_of)
            / "inference"
            / self.model_fingerprint
        )
        cached_path = inference_root / "api_predictions.json"
        prediction_map: dict[str, dict] = {}
        if cached_path.exists():
            payload = json.loads(cached_path.read_text(encoding="utf-8"))
            prediction_map = {
                item["period"]: item["prediction"] for item in payload["results"]
            }

        missing = [
            result for result in valid if result["period"] not in prediction_map
        ]
        if not missing:
            return prediction_map

        manifest_rows = []
        for result in missing:
            manifest_rows.append(
                {
                    "rank": stock.get("rank", 0),
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "frequency": result["period"],
                    "periodStart": result["periodStart"],
                    "periodEnd": result["periodEnd"],
                    "sourceBars": result["sourceBars"],
                    "candles": result["candles"],
                    "imagePath": result["imagePath"],
                    "candlePath": result["candlePath"],
                    "status": "ok",
                    "error": "",
                }
            )

        predictions, _, _ = pipeline.run_pytorch_inference(
            self.model,
            pd.DataFrame(manifest_rows),
            inference_root,
            config=self.inference_config,
            class_names=self.class_names,
            frequencies=[result["period"] for result in missing],
        )
        for row in predictions.itertuples(index=False):
            prediction = {
                "predictedIndex": int(row.predictedIndex),
                "predictedLabel": str(row.predictedLabel),
                "confidence": float(row.confidence),
                "topK": json.loads(row.topK),
            }
            prediction_map[str(row.frequency)] = prediction
        response_results = [
            {"period": period, "prediction": prediction_map[period]}
            for period in sorted(prediction_map)
        ]
        _atomic_json(
            {
                "modelFingerprint": self.model_fingerprint,
                "results": response_results,
            },
            cached_path,
        )
        return prediction_map

    def _prediction_cache_path(self, symbol: str, as_of: date) -> Path:
        return (
            self._request_root(symbol, as_of)
            / "inference"
            / self.model_fingerprint
            / "api_predictions.json"
        )

    def _prediction_cache_covers(
        self,
        symbol: str,
        as_of: date,
        periods: list[str],
    ) -> bool:
        path = self._prediction_cache_path(symbol, as_of)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            cached_periods = {item["period"] for item in payload["results"]}
            return set(periods).issubset(cached_periods)
        except Exception:
            return False

    def analyze(self, request: AnalyzeRequest) -> dict:
        """요청 하나를 동기적으로 처리합니다. FastAPI가 thread pool에서 호출합니다."""
        periods = [int(value) for value in request.periods]
        self._validate_request_date(request.as_of, periods)
        if request.predict and self.model is None:
            raise RuntimeError(
                "CNN 모델이 설정되지 않았습니다. MODEL_PATH를 설정하거나 "
                "predict=false로 요청하세요."
            )

        # matplotlib 렌더링, 토스 Session, GPU 모델을 여러 스레드가 동시에 건드리지
        # 않도록 단일 프로세스 안에서는 요청 처리 구간을 직렬화합니다.
        with self._operation_lock:
            stock = self._resolve_stock(request.symbol)
            cached_before = []
            missing_periods = []
            for months in periods:
                paths = self._period_paths(request.symbol, request.as_of, months)
                cached_result = self._load_period_cache(paths)
                cached_before.append(cached_result is not None)
                if cached_result is None:
                    missing_periods.append(months)

            history = pd.DataFrame(columns=pipeline.OHLC_COLUMNS)
            if missing_periods:
                earliest_start = min(
                    rolling_window_start(request.as_of, value)
                    for value in missing_periods
                )
                history = self._load_or_fetch_history(
                    request.symbol, earliest_start, request.as_of
                )

            period_results = []
            for months in periods:
                result, _ = self._build_period(
                    stock, request.as_of, months, history
                )
                period_results.append(result)

            valid_results = [
                result for result in period_results if result["status"] == "ok"
            ]
            prediction_cached_before = (
                not request.predict
                or not valid_results
                or self._prediction_cache_covers(
                    request.symbol,
                    request.as_of,
                    [result["period"] for result in valid_results],
                )
            )
            predictions = (
                self._run_inference(stock, request.as_of, period_results)
                if request.predict
                else {}
            )
            for result in period_results:
                result["prediction"] = predictions.get(result["period"])

            request_key = (
                f"{request.symbol}|{request.as_of}|{','.join(map(str, periods))}|"
                f"adjusted={self.settings.adjusted}|predict={request.predict}|"
                f"model={self.model_fingerprint}"
            )
            return {
                "requestId": hashlib.sha256(request_key.encode("utf-8")).hexdigest()[:16],
                "symbol": stock["symbol"],
                "name": stock["name"],
                "market": stock["market"],
                "asOf": request.as_of.isoformat(),
                "requestedPeriods": [f"{value}M" for value in periods],
                "targetCandles": 100,
                "adjusted": self.settings.adjusted,
                "cached": all(cached_before) and prediction_cached_before,
                "results": period_results,
            }

    def image_path(self, symbol: str, as_of: date, period: str) -> Path:
        months = self._parse_period(period)
        return self._period_paths(symbol, as_of, months)["image"]

    def candle_path(self, symbol: str, as_of: date, period: str) -> Path:
        months = self._parse_period(period)
        return self._period_paths(symbol, as_of, months)["candles"]

    @staticmethod
    def _parse_period(value: str) -> int:
        normalized = value.upper()
        if normalized not in {"6M", "12M"}:
            raise ValueError("period는 6M 또는 12M이어야 합니다.")
        return int(normalized[:-1])


def _file_url(
    request: Request,
    route_name: str,
    settings: APISettings,
    *,
    symbol: str,
    as_of: date,
    period: str,
) -> str:
    relative = str(
        request.url_for(
            route_name,
            symbol=symbol,
            as_of=as_of.isoformat(),
            period=period,
        )
    )
    if not settings.public_base_url:
        return relative
    # 리버스 프록시가 내부 호스트를 전달하는 환경에서는 고정 공개 주소로 교체합니다.
    path = request.app.url_path_for(
        route_name,
        symbol=symbol,
        as_of=as_of.isoformat(),
        period=period,
    )
    return settings.public_base_url.rstrip("/") + str(path)


def create_app(
    settings: APISettings | None = None,
    service: OnDemandChartService | None = None,
) -> FastAPI:
    settings = settings or APISettings.from_env()
    app = FastAPI(
        title="KOSPI Rolling Chart CNN API",
        version="1.0.0",
        description=(
            "종목코드와 기준일을 받아 이전 6/12개월의 실제 일봉을 "
            "100캔들 흑백 이미지로 만들고 선택적으로 CNN을 실행합니다."
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

    def get_service() -> OnDemandChartService:
        if app.state.service is None:
            # 첫 요청 시 한 번만 토스 클라이언트와 모델을 로드합니다.
            with app.state.service_init_lock:
                if app.state.service is None:
                    try:
                        app.state.service = OnDemandChartService(settings)
                    except Exception as error:
                        LOGGER.exception("API 서버 초기 설정 실패")
                        raise HTTPException(
                            status_code=503,
                            detail=(
                                "API 서버 설정을 불러오지 못했습니다. 토스 키, 모델 경로, "
                                "클래스 매핑 및 저장 권한을 확인하세요."
                            ),
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
    def health(service_instance: OnDemandChartService = Depends(get_service)) -> dict:
        return {
            "status": "ok",
            "modelLoaded": service_instance.model_loaded,
            "modelFingerprint": service_instance.model_fingerprint,
            "universeRestricted": service_instance._universe is not None,
        }

    @app.post(
        "/v1/analyze",
        response_model=AnalyzeResponse,
        response_model_by_alias=True,
        dependencies=[Depends(require_api_key)],
    )
    def analyze(
        payload: AnalyzeRequest,
        web_request: Request,
        service_instance: OnDemandChartService = Depends(get_service),
    ) -> dict:
        try:
            response = service_instance.analyze(payload)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except pipeline.TossAPIError as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except Exception as error:
            LOGGER.exception("온디맨드 분석 중 예기치 않은 오류")
            raise HTTPException(
                status_code=500, detail="서버 내부 분석 오류가 발생했습니다."
            ) from error

        for result in response["results"]:
            if result["status"] != "ok":
                continue
            result["imageUrl"] = _file_url(
                web_request,
                "get_image",
                settings,
                symbol=response["symbol"],
                as_of=payload.as_of,
                period=result["period"],
            )
            result["candleUrl"] = _file_url(
                web_request,
                "get_candles",
                settings,
                symbol=response["symbol"],
                as_of=payload.as_of,
                period=result["period"],
            )
        return response

    @app.get(
        "/v1/images/{symbol}/{as_of}/{period}.png",
        name="get_image",
    )
    def get_image(
        symbol: str,
        as_of: date,
        period: str,
        service_instance: OnDemandChartService = Depends(get_service),
    ) -> FileResponse:
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise HTTPException(status_code=422, detail="잘못된 종목코드입니다.")
        try:
            path = service_instance.image_path(symbol, as_of, period)
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
        "/v1/candles/{symbol}/{as_of}/{period}.csv",
        name="get_candles",
    )
    def get_candles(
        symbol: str,
        as_of: date,
        period: str,
        service_instance: OnDemandChartService = Depends(get_service),
    ) -> FileResponse:
        if not SYMBOL_PATTERN.fullmatch(symbol):
            raise HTTPException(status_code=422, detail="잘못된 종목코드입니다.")
        try:
            path = service_instance.candle_path(symbol, as_of, period)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        if not path.exists():
            raise HTTPException(status_code=404, detail="생성된 캔들 CSV가 없습니다.")
        return FileResponse(
            path,
            media_type="text/csv; charset=utf-8",
            filename=f"{symbol}_{as_of.isoformat()}_{period.upper()}_100candles.csv",
        )

    return app


app = create_app()
