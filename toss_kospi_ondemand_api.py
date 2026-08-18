"""기존 Render Start Command를 유지하기 위한 호환 진입점.

실제 구현은 ``kospi_chart_render_api.py``에 있으며 토스증권을 직접 호출하지 않습니다.
"""

from kospi_chart_render_api import app

__all__ = ["app"]
