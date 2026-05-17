import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from backend.core.config import get_settings


def main() -> int:
    settings = get_settings()
    health_url = settings.fish_speech_url.rstrip("/").removesuffix("/v1/tts") + "/v1/health"
    try:
        response = requests.get(health_url, timeout=5)
    except requests.RequestException as exc:
        print(f"Fish Speech server is not reachable at {health_url}: {exc}")
        return 1
    if response.status_code != 200:
        print(f"Fish Speech health check failed: {response.status_code} {response.text[:300]}")
        return 1
    print(f"Fish Speech server is ready: {response.text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
