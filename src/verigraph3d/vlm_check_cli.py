from __future__ import annotations

import argparse
import json

from .vlm import VLMImage, VLMRequest, VLMSettings, create_vlm_client


CHECK_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "message": {"type": "string"}},
    "required": ["ok", "message"],
    "additionalProperties": False,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate VeriGraph3D VLM configuration")
    parser.add_argument("--live", action="store_true", help="Send one billable health-check request")
    parser.add_argument("--image", help="Optional image for a live multimodal check")
    args = parser.parse_args(argv)
    settings = VLMSettings.from_env()
    safe = {
        "provider": settings.provider,
        "base_url": settings.base_url,
        "model": settings.model,
        "api_key_configured": bool(settings.api_key),
        "structured_outputs": settings.structured_outputs,
        "max_retries": settings.max_retries,
    }
    if not args.live:
        print(json.dumps({"configuration": safe, "live_request_sent": False}, ensure_ascii=False, indent=2))
        return 0
    client = create_vlm_client(settings)
    response = client.complete(VLMRequest(
        system="Return the requested health-check JSON only.",
        prompt="Return ok=true and a short message confirming text input" + (" and image input" if args.image else "") + ".",
        images=(VLMImage(args.image),) if args.image else (),
        json_schema=CHECK_SCHEMA,
        schema_name="verigraph3d_health_check",
        max_output_tokens=128,
    ))
    print(json.dumps({"configuration": safe, "live_request_sent": True, "response": response.data, "usage": {"calls": client.usage.calls, "input_tokens": client.usage.input_tokens, "output_tokens": client.usage.output_tokens}}, ensure_ascii=False, indent=2))
    return 0 if response.data.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
