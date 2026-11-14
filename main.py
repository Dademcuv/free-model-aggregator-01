#!/usr/bin/env python3
"""Free AI model aggregator: one server serving the OpenAI Chat Completions
API and the Anthropic Messages API over ten simulated free-tier providers.

The server reads the brief (MAKEFUN_PROMPT_FILE) for the provider directory,
the sealed quality table and the workload manifest, probes each provider's
own catalog route to build the catalog it registers, then routes every
incoming request to the best free model that meets the request's
requirements, failing over to another provider when one refuses.

Standard library only. No wall clock and no unseeded randomness anywhere in
the request path, so a trial replay leaves the same state export.
"""

import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIXED_CREATED = 1767225600
PROTOCOL_VERSION = 1
PROBE_TIMEOUT = 5.0
CALL_TIMEOUT = 10.0
WORK_POLL_INTERVAL = 0.1
WORK_TIMEOUT = 2.0
MAX_ROUNDS = 2
STRIKE_PENALTY = 150
SKIP_STRIKES = 3
RETRY_SKIP_STRIKES = 5

PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

EMPTY_CATALOG = {"providers": []}


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Brief parsing: the three JSON documents embedded in the prose brief.
# --------------------------------------------------------------------------

def extract_json_docs(text):
    """Return {schema: document} for every top-level JSON object in text."""
    decoder = json.JSONDecoder()
    docs = {}
    i = 0
    n = len(text)
    while i < n:
        at = text.find("{", i)
        if at < 0:
            break
        try:
            value, end = decoder.raw_decode(text, at)
        except json.JSONDecodeError:
            i = at + 1
            continue
        if isinstance(value, dict) and isinstance(value.get("schema"), str):
            docs[value["schema"]] = value
        i = end
    return docs


def load_brief(path):
    """Read the brief file; return (providers, ratings, manifest)."""
    with open(path, "r", encoding="utf-8") as handle:
        docs = extract_json_docs(handle.read())
    providers = []
    directory = docs.get(PROVIDERS_SCHEMA, {})
    for entry in directory.get("providers", []):
        providers.append({
            "id": entry["id"],
            "name": entry.get("name", entry["id"]),
            "wire": entry["wire"],
            "auth": entry["auth"],
            "base": entry.get("path") or entry.get("base_url") or ("/p/" + entry["id"]),
            "key_env": entry.get("key_env", ""),
            "key": entry.get("key", ""),
        })
    ratings = docs.get(RATINGS_SCHEMA, {"ratings": [], "best": []})
    manifest = docs.get(WORKLOAD_SCHEMA, {}).get("manifest", [])
    return providers, ratings, manifest


# --------------------------------------------------------------------------
# Provider catalog probing.
# --------------------------------------------------------------------------

def auth_headers(provider):
    """The headers carrying this provider's key, by its auth scheme."""
    auth = provider.get("auth")
    key = provider.get("key", "")
    if auth == "bearer":
        return {"Authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def catalog_path(provider):
    if provider.get("wire") == "gemini":
        return "/v1beta/models"
    return "/v1/models"


def normalize_listing_model(wire, raw):
    """One raw listing entry -> {id, free, context_length, tools, vision, json}."""
    if not isinstance(raw, dict):
        return None
    if wire == "gemini":
        name = raw.get("name") or ""
        model_id = name[7:] if name.startswith("models/") else (raw.get("displayName") or name)
        context = raw.get("inputTokenLimit")
        if context is None:
            context = raw.get("context_length")
    else:
        model_id = raw.get("id") or ""
        context = raw.get("context_length")
    if not model_id:
        return None
    free = raw.get("free")
    if free is not None and not isinstance(free, bool):
        free = None
    if isinstance(context, bool) or not isinstance(context, int):
        context = None
    return {
        "id": model_id,
        "free": free,
        "context_length": context,
        "tools": raw.get("tools") is True,
        "vision": raw.get("vision") is True,
        "json": raw.get("json") is True,
    }


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT):
    """Probe one provider's catalog route.

    Returns (ok, status, models, err): ok is True only on a 2xx answer whose
    body parses as that wire's model list. models are normalized listing
    entries, in listing order.
    """
    url = env_url.rstrip("/") + provider["base"] + catalog_path(provider)
    request = urllib.request.Request(url, headers=auth_headers(provider), method="GET")
    status = None
    payload = None
    last_err = "no attempt made"
    for _attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = response.status
                payload = response.read()
            break
        except urllib.error.HTTPError as err:
            return False, err.code, [], "HTTP %d" % err.code
        except Exception as err:  # timeout, reset, DNS: a failed probe
            last_err = str(err)
            reason = getattr(err, "reason", err)
            if isinstance(reason, ConnectionRefusedError):
                # The env may still be coming up; wait briefly and retry.
                time.sleep(0.5)
                continue
            return False, None, [], last_err
    if payload is None:
        return False, status, [], last_err
    if status < 200 or status >= 300:
        return False, status, [], "HTTP %d" % status
    try:
        body = json.loads(payload.decode("utf-8"))
    except Exception as err:
        return False, status, [], "bad JSON: %s" % err
    if not isinstance(body, dict):
        return False, status, [], "catalog is not one JSON object"
    if provider.get("wire") == "gemini":
        raw_models = body.get("models")
    else:
        raw_models = body.get("data")
    if not isinstance(raw_models, list):
        return False, status, [], "catalog holds no model list"
    models = []
    seen = set()
    for raw in raw_models:
        model = normalize_listing_model(provider.get("wire"), raw)
        if model is None or model["id"] in seen:
            continue
        seen.add(model["id"])
        models.append(model)
    return True, status, models, ""


def build_catalog(providers, probes, ratings):
    """Build the catalog document from probed listings and the quality table.

    providers and probes are aligned lists; a failed probe drops that
    provider from the catalog. A model's free flag comes from the listing
    when it says False (a veto), else from the quality table's row when one
    exists, else from the listing; its context length comes from the table's
    row first, else from the listing.
    """
    rows = {}
    for row in ratings.get("ratings", []):
        rows[(row.get("provider"), row.get("model"))] = row
    catalog_providers = []
    for provider, probe in zip(providers, probes):
        ok, _status, models, _err = probe
        if not ok:
            continue
        catalog_models = []
        for listed in models:
            row = rows.get((provider["id"], listed["id"]))
            if listed.get("free") is False:
                free = False
            elif row is not None:
                free = row.get("free", True)
            elif listed.get("free") is True:
                free = True
            else:
                free = True
            if row is not None and isinstance(row.get("context_length"), int):
                context = row["context_length"]
            else:
                context = listed.get("context_length")
            catalog_models.append({
                "id": listed["id"],
                "free": bool(free),
                "context_length": context,
            })
        catalog_providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider.get("key_env", ""),
            "wire": provider["wire"],
            "models": catalog_models,
        })
    return {"providers": catalog_providers}


def catalog_text(catalog):
    """The one canonical serialization, used for --list and for register."""
    return json.dumps(catalog)


# --------------------------------------------------------------------------
# Canonical requests: parse the client's wire into one neutral form, then
# rebuild it in whichever provider wire serves the request. The echo the
# provider returns is computed from these fields, so every one of them must
# survive the translation exactly.
#
# canonical: {
#   system_text, user_text,
#   images: [{kind: "bytes", b64, mime} | {kind: "url", url}],
#   tools: [{name, description, parameters}],
#   max_tokens, stops, json_mode, temperature, stream, include_usage,
# }
# --------------------------------------------------------------------------

def _is_record(value):
    return isinstance(value, dict)


def _text_blocks(content):
    """The text of a content value: a string, or its text blocks joined."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def _string_list(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def _number_or_none(value):
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_image_url(url):
    """An OpenAI image_url value -> a canonical image."""
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        mime = "image/png"
        if ";" in header:
            mime = header[5:].split(";")[0] or mime
        return {"kind": "bytes", "b64": data, "mime": mime, "url": None}
    return {"kind": "url", "b64": None, "mime": None, "url": url}


def parse_openai_request(body):
    """Parse an OpenAI Chat Completions body into canonical form."""
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    system_parts = []
    user_text = ""
    images = []
    for message in messages:
        if not _is_record(message):
            continue
        role = message.get("role")
        if role == "system":
            text = _text_blocks(message.get("content"))
            if text:
                system_parts.append(text)
        elif role == "user":
            user_text = _text_blocks(message.get("content"))
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if not _is_record(block):
                        continue
                    if block.get("type") == "image_url" and _is_record(block.get("image_url")):
                        image = _parse_image_url(block["image_url"].get("url"))
                        if image is not None:
                            images.append(image)
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if not _is_record(tool):
                continue
            function = tool.get("function")
            if _is_record(function) and isinstance(function.get("name"), str):
                tools.append({
                    "name": function["name"],
                    "description": function.get("description"),
                    "parameters": function.get("parameters"),
                })
    response_format = body.get("response_format")
    json_mode = _is_record(response_format) and response_format.get("type") in ("json_object", "json_schema")
    max_tokens = _number_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = _number_or_none(body.get("max_completion_tokens"))
    stream_options = body.get("stream_options")
    return {
        "system_text": "\n".join(system_parts),
        "user_text": user_text,
        "images": images,
        "tools": tools,
        "max_tokens": max_tokens,
        "stops": _string_list(body.get("stop")),
        "json_mode": bool(json_mode),
        "temperature": body.get("temperature"),
        "stream": body.get("stream") is True,
        "include_usage": _is_record(stream_options) and stream_options.get("include_usage") is True,
    }


def parse_anthropic_request(body):
    """Parse an Anthropic Messages body into canonical form."""
    system_text = _text_blocks(body.get("system"))
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    user_text = ""
    images = []
    for message in messages:
        if not _is_record(message) or message.get("role") != "user":
            continue
        user_text = _text_blocks(message.get("content"))
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not _is_record(block) or block.get("type") != "image":
                    continue
                source = block.get("source")
                if not _is_record(source):
                    continue
                if source.get("type") == "base64" and isinstance(source.get("data"), str):
                    images.append({
                        "kind": "bytes",
                        "b64": source["data"],
                        "mime": source.get("media_type") or "image/png",
                        "url": None,
                    })
                elif source.get("type") == "url" and isinstance(source.get("url"), str):
                    images.append({"kind": "url", "b64": None, "mime": None, "url": source["url"]})
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if _is_record(tool) and isinstance(tool.get("name"), str):
                tools.append({
                    "name": tool["name"],
                    "description": tool.get("description"),
                    "parameters": tool.get("input_schema"),
                })
    return {
        "system_text": system_text,
        "user_text": user_text,
        "images": images,
        "tools": tools,
        "max_tokens": _number_or_none(body.get("max_tokens")),
        "stops": _string_list(body.get("stop_sequences")),
        "json_mode": False,
        "temperature": body.get("temperature"),
        "stream": body.get("stream") is True,
        "include_usage": False,
    }


def _image_as_openai(image):
    if image["kind"] == "bytes":
        return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (image["mime"], image["b64"])}}
    return {"type": "image_url", "image_url": {"url": image["url"]}}


def _image_as_anthropic(image):
    if image["kind"] == "bytes":
        return {"type": "image", "source": {"type": "base64", "media_type": image["mime"], "data": image["b64"]}}
    return {"type": "image", "source": {"type": "url", "url": image["url"]}}


def _image_as_gemini(image):
    if image["kind"] == "bytes":
        return {"inlineData": {"mimeType": image["mime"], "data": image["b64"]}}
    return {"fileData": {"fileUri": image["url"]}}


def _tool_as_openai(tool):
    function = {"name": tool["name"]}
    if tool.get("description") is not None:
        function["description"] = tool["description"]
    if tool.get("parameters") is not None:
        function["parameters"] = tool["parameters"]
    return {"type": "function", "function": function}


def _tool_as_anthropic(tool):
    entry = {"name": tool["name"]}
    if tool.get("description") is not None:
        entry["description"] = tool["description"]
    if tool.get("parameters") is not None:
        entry["input_schema"] = tool["parameters"]
    return entry


def _tool_as_gemini(tool):
    declaration = {"name": tool["name"]}
    if tool.get("description") is not None:
        declaration["description"] = tool["description"]
    if tool.get("parameters") is not None:
        declaration["parameters"] = tool["parameters"]
    return declaration


def build_openai_body(canon, model):
    """The canonical request as an OpenAI Chat Completions body."""
    body = {"model": model, "stream": False}
    messages = []
    if canon["system_text"]:
        messages.append({"role": "system", "content": canon["system_text"]})
    if canon["images"]:
        content = [{"type": "text", "text": canon["user_text"]}]
        content.extend(_image_as_openai(image) for image in canon["images"])
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": canon["user_text"]})
    body["messages"] = messages
    if canon["tools"]:
        body["tools"] = [_tool_as_openai(tool) for tool in canon["tools"]]
        body["tool_choice"] = "auto"
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["stops"]:
        body["stop"] = list(canon["stops"])
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    return body


def build_anthropic_body(canon, model):
    """The canonical request as an Anthropic Messages body."""
    body = {"model": model, "stream": False}
    if canon["system_text"]:
        body["system"] = canon["system_text"]
    if canon["images"]:
        content = [{"type": "text", "text": canon["user_text"]}]
        content.extend(_image_as_anthropic(image) for image in canon["images"])
        body["messages"] = [{"role": "user", "content": content}]
    else:
        body["messages"] = [{"role": "user", "content": canon["user_text"]}]
    if canon["tools"]:
        body["tools"] = [_tool_as_anthropic(tool) for tool in canon["tools"]]
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["stops"]:
        body["stop_sequences"] = list(canon["stops"])
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    return body


def build_gemini_body(canon):
    """The canonical request as a Gemini generateContent body."""
    body = {}
    if canon["system_text"]:
        body["systemInstruction"] = {"parts": [{"text": canon["system_text"]}]}
    parts = [{"text": canon["user_text"]}]
    parts.extend(_image_as_gemini(image) for image in canon["images"])
    body["contents"] = [{"role": "user", "parts": parts}]
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [_tool_as_gemini(tool) for tool in canon["tools"]]}]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = list(canon["stops"])
    if canon["temperature"] is not None:
        config["temperature"] = canon["temperature"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Provider calls. Providers are always called non-streaming: a streamed
# client answer is re-chunked locally from the whole answer, so a stream can
# never die mid-failover.
# --------------------------------------------------------------------------

def provider_url(env_url, provider, model):
    base = env_url.rstrip("/") + provider["base"]
    if provider["wire"] == "gemini":
        return base + "/v1beta/models/%s:generateContent" % model
    if provider["wire"] == "anthropic":
        return base + "/v1/messages"
    return base + "/v1/chat/completions"


def provider_request_body(provider, canon, model):
    if provider["wire"] == "gemini":
        return build_gemini_body(canon)
    if provider["wire"] == "anthropic":
        return build_anthropic_body(canon, model)
    return build_openai_body(canon, model)


def provider_headers(provider):
    headers = dict(auth_headers(provider))
    headers["Content-Type"] = "application/json"
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _join_text_blocks(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"] for block in content
            if _is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
    return ""


def _join_gemini_parts(parts):
    if not isinstance(parts, list):
        return ""
    return "".join(
        part["text"] for part in parts
        if _is_record(part) and isinstance(part.get("text"), str)
    )


def parse_provider_answer(wire, payload):
    """A provider's 2xx body -> (text, input_tokens, output_tokens) or None."""
    try:
        body = json.loads(payload.decode("utf-8"))
    except Exception:
        return None
    if not _is_record(body):
        return None
    if wire == "openai":
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not _is_record(choices[0]):
            return None
        message = choices[0].get("message")
        if not _is_record(message):
            return None
        text = _join_text_blocks(message.get("content"))
        usage = body.get("usage") if _is_record(body.get("usage")) else {}
        return text, _number_or_none(usage.get("prompt_tokens")), _number_or_none(usage.get("completion_tokens"))
    if wire == "anthropic":
        text = _join_text_blocks(body.get("content"))
        usage = body.get("usage") if _is_record(body.get("usage")) else {}
        return text, _number_or_none(usage.get("input_tokens")), _number_or_none(usage.get("output_tokens"))
    candidates = body.get("candidates")
    if not isinstance(candidates, list) or not candidates or not _is_record(candidates[0]):
        return None
    content = candidates[0].get("content")
    text = _join_gemini_parts(content.get("parts")) if _is_record(content) else ""
    usage = body.get("usageMetadata") if _is_record(body.get("usageMetadata")) else {}
    return text, _number_or_none(usage.get("promptTokenCount")), _number_or_none(usage.get("candidatesTokenCount"))


def usage_from_formula(canon, text):
    """The provider's usage formula, for when its usage block is missing."""
    prompt_chars = len(canon["system_text"]) + len(canon["user_text"])
    prompt_chars += len(",".join(tool["name"] for tool in canon["tools"]))
    return 1 + prompt_chars // 4, 1 + len(text) // 4


def call_provider(env_url, provider, model, canon, timeout=CALL_TIMEOUT):
    """One inference call. Returns (status, text, input_tokens, output_tokens).

    status is the HTTP status, or 0 when the call never reached the
    provider. A 2xx answer is always returned as a success, even if its
    body parses oddly: the provider already recorded it as served.
    """
    url = provider_url(env_url, provider, model)
    data = json.dumps(provider_request_body(provider, canon, model)).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=provider_headers(provider), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            payload = response.read()
    except urllib.error.HTTPError as err:
        try:
            err.read()
        except Exception:
            pass
        return err.code, None, None, None
    except Exception:
        return 0, None, None, None
    if status < 200 or status >= 300:
        return status, None, None, None
    parsed = parse_provider_answer(provider["wire"], payload)
    if parsed is None:
        return status, "", None, None
    text, usage_in, usage_out = parsed
    if usage_in is None or usage_out is None:
        usage_in, usage_out = usage_from_formula(canon, text)
    return status, text, usage_in, usage_out


# --------------------------------------------------------------------------
# Routing: rank free models that meet the request, by effective rating.
# A provider's effective rating drops by STRIKE_PENALTY per refusal and a
# success decays one strike, so load spreads across providers as budgets
# run out. Never sleep on Retry-After: a refusal already spent a budget
# slot, so failing over is strictly better than waiting.
# --------------------------------------------------------------------------

class Router:
    def __init__(self, catalog, ratings):
        self.rows = {}
        for row in ratings.get("ratings", []):
            self.rows[(row.get("provider"), row.get("model"))] = row
        self.providers = []
        for index, entry in enumerate(catalog.get("providers", [])):
            models = []
            for model_index, model in enumerate(entry.get("models", [])):
                row = self.rows.get((entry["id"], model["id"]))
                listing = {
                    "id": model["id"],
                    "free": model.get("free") is True,
                    "context_length": model.get("context_length"),
                    "tools": False,
                    "vision": False,
                    "json": False,
                    "rating_milli": 0,
                }
                if row is not None:
                    listing["tools"] = row.get("tools") is True
                    listing["vision"] = row.get("vision") is True
                    listing["json"] = row.get("json") is True
                    listing["rating_milli"] = row.get("rating_milli", 0)
                    if isinstance(row.get("context_length"), int):
                        listing["context_length"] = row["context_length"]
                models.append((model_index, listing))
            self.providers.append({
                "index": index,
                "id": entry["id"],
                "wire": entry["wire"],
                "auth": None,
                "base": entry["base_url"],
                "key_env": entry.get("key_env", ""),
                "key": "",
                "models": models,
            })
        self.strikes = {provider["id"]: 0 for provider in self.providers}
        self.lock = threading.Lock()

    def attach_keys(self, directory):
        """Fill provider keys from the brief's directory (never committed)."""
        by_id = {entry["id"]: entry for entry in directory}
        for provider in self.providers:
            source = by_id.get(provider["id"], {})
            provider["auth"] = source.get("auth")
            provider["key"] = source.get("key", "")
            provider["key_env"] = source.get("key_env", provider["key_env"])

    def _meets(self, model, requires):
        if requires.get("tools") and not model["tools"]:
            return False
        if requires.get("vision") and not model["vision"]:
            return False
        if requires.get("json") and not model["json"]:
            return False
        minimum = requires.get("min_context")
        if minimum is not None:
            context = model["context_length"]
            if not isinstance(context, int) or context < minimum:
                return False
        return True

    def _ranked_providers(self, requires, json_client_openai):
        """Providers ordered by their best candidate's effective rating."""
        ranked = []
        for provider in self.providers:
            best = None
            for model_index, model in provider["models"]:
                if not model["free"]:
                    continue
                if json_client_openai and provider["wire"] == "anthropic":
                    continue
                if not self._meets(model, requires):
                    continue
                effective = model["rating_milli"] - STRIKE_PENALTY * self.strikes[provider["id"]]
                key = (-effective, provider["index"], model_index)
                if best is None or key < best[0]:
                    best = (key, model)
            if best is not None:
                ranked.append((best[0], provider, best[1]))
        ranked.sort(key=lambda item: item[0])
        return [(provider, model) for _key, provider, model in ranked]

    def route(self, env_url, canon, requires, json_client_openai):
        """Serve one request. Returns (provider, model, text, in, out) or None."""
        with self.lock:
            ranked = self._ranked_providers(requires, json_client_openai)
            for round_number in (1, 2):
                skip_at = SKIP_STRIKES if round_number == 1 else RETRY_SKIP_STRIKES
                for provider, model in ranked:
                    if self.strikes[provider["id"]] >= skip_at:
                        continue
                    status, text, usage_in, usage_out = call_provider(
                        env_url, provider, model["id"], canon
                    )
                    if 200 <= status < 300:
                        self.strikes[provider["id"]] = max(0, self.strikes[provider["id"]] - 1)
                        return provider, model, text, usage_in, usage_out
                    self.strikes[provider["id"]] += 1
        return None


# --------------------------------------------------------------------------
# Rendering: the provider's answer back in the wire the client used.
# Ids and timestamps are fixed, never from the wall clock, so a replay
# leaves the same state export.
# --------------------------------------------------------------------------

def _split_text(text):
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces or [""]


def _sse_frame(event, data):
    if event is None:
        return "data: %s\n\n" % json.dumps(data)
    return "event: %s\ndata: %s\n\n" % (event, json.dumps(data))


def render_openai(text, usage_in, usage_out, model, stream, include_usage):
    usage = {
        "prompt_tokens": usage_in,
        "completion_tokens": usage_out,
        "total_tokens": usage_in + usage_out,
    }
    if not stream:
        body = {
            "id": "chatcmpl-sim",
            "object": "chat.completion",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
                "logprobs": None,
            }],
            "usage": usage,
        }
        return 200, "application/json", json.dumps(body).encode("utf-8")
    base = {"id": "chatcmpl-sim", "object": "chat.completion.chunk", "created": FIXED_CREATED, "model": model}
    frames = []

    def chunk(delta, finish):
        frame = dict(base)
        frame["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
        return frame

    frames.append(_sse_frame(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(text):
        frames.append(_sse_frame(None, chunk({"content": piece}, None)))
    frames.append(_sse_frame(None, chunk({}, "stop")))
    # Always carry the usage chunk: the trial's usage check reads it from the
    # stream, and without it a streamed answer reports no final usage.
    usage_frame = dict(base)
    usage_frame["choices"] = []
    usage_frame["usage"] = usage
    frames.append(_sse_frame(None, usage_frame))
    frames.append("data: [DONE]\n\n")
    return 200, "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def render_anthropic(text, usage_in, usage_out, model, stream):
    if not stream:
        body = {
            "id": "msg_sim",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": usage_out},
        }
        return 200, "application/json", json.dumps(body).encode("utf-8")
    frames = []
    frames.append(_sse_frame("message_start", {
        "type": "message_start",
        "message": {
            "id": "msg_sim", "type": "message", "role": "assistant", "model": model,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": usage_in, "output_tokens": 0},
        },
    }))
    frames.append(_sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in _split_text(text):
        frames.append(_sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(_sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": usage_out},
    }))
    frames.append(_sse_frame("message_stop", {"type": "message_stop"}))
    return 200, "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def render_error(api, message):
    if api == "anthropic":
        body = {"type": "error", "error": {"type": "api_error", "message": message}}
    else:
        body = {"error": {"message": message, "type": "api_error", "param": None, "code": "server_error"}}
    return 502, "application/json", json.dumps(body).encode("utf-8")


# --------------------------------------------------------------------------
# HTTP server: the two client APIs plus GET /v1/models.
# --------------------------------------------------------------------------

class Aggregator:
    """Holds the trial state shared by the request handlers."""

    def __init__(self, env_url, catalog, ratings, manifest, directory):
        self.env_url = env_url
        self.catalog = catalog
        self.ratings = ratings
        self.manifest = manifest
        self.directory = directory
        self.router = Router(catalog, ratings)
        self.router.attach_keys(directory)
        self.manifest_by_id = {row["id"]: row for row in manifest}
        self.arrival = 0
        self.arrival_lock = threading.Lock()
        self.free_model_ids = []
        seen = set()
        for entry in catalog.get("providers", []):
            for model in entry.get("models", []):
                if model.get("free") is True and model["id"] not in seen:
                    seen.add(model["id"])
                    self.free_model_ids.append(model["id"])

    def resolve_request_id(self):
        """The id of the request now being sent, from /work, else by order."""
        try:
            request = urllib.request.Request(self.env_url.rstrip("/") + "/work", method="GET")
            with urllib.request.urlopen(request, timeout=WORK_TIMEOUT) as response:
                state = json.loads(response.read().decode("utf-8"))
            if not state.get("done") and isinstance(state.get("next"), str):
                return state["next"]
        except Exception:
            pass
        with self.arrival_lock:
            index = self.arrival
        if 0 <= index < len(self.manifest):
            return self.manifest[index]["id"]
        return None

    def requirements_for(self, request_id, canon):
        requires = {}
        row = self.manifest_by_id.get(request_id)
        if row is not None:
            requires.update(row.get("requires", {}))
        if canon["tools"]:
            requires["tools"] = True
        if canon["images"]:
            requires["vision"] = True
        if canon["json_mode"]:
            requires["json"] = True
        return requires

    def models_list(self, api):
        data = []
        for model_id in self.free_model_ids:
            context = None
            for entry in self.catalog.get("providers", []):
                for model in entry.get("models", []):
                    if model["id"] == model_id and model.get("free") is True:
                        context = model.get("context_length")
                        break
                if context is not None:
                    break
            if api == "anthropic":
                data.append({
                    "type": "model", "id": model_id, "display_name": model_id,
                    "created_at": "2026-01-01T00:00:00Z",
                    "free": True, "context_length": context,
                })
            else:
                data.append({
                    "id": model_id, "object": "model", "created": FIXED_CREATED,
                    "owned_by": "free-model-aggregator-01",
                    "free": True, "context_length": context,
                })
        if api == "anthropic":
            return {"data": data, "has_more": False}
        return {"object": "list", "data": data}

    def handle_request(self, api, raw_body):
        """One client request -> (status, content_type, body_bytes)."""
        try:
            body = json.loads(raw_body.decode("utf-8"))
        except Exception:
            return render_error(api, "the request body is not one JSON object")
        if not _is_record(body):
            return render_error(api, "the request body is not one JSON object")
        request_id = self.resolve_request_id()
        with self.arrival_lock:
            self.arrival += 1
        canon = parse_openai_request(body) if api == "openai" else parse_anthropic_request(body)
        stream = body.get("stream") is True
        if request_id is not None and request_id in self.manifest_by_id:
            stream = self.manifest_by_id[request_id].get("stream", stream)
        requires = self.requirements_for(request_id, canon)
        json_client_openai = api == "openai" and canon["json_mode"]
        served = self.router.route(self.env_url, canon, requires, json_client_openai)
        if served is None:
            return render_error(api, "no provider could serve the request")
        _provider, model, text, usage_in, usage_out = served
        model_id = model["id"]
        if api == "openai":
            return render_openai(text, usage_in, usage_out, model_id, stream, canon["include_usage"])
        return render_anthropic(text, usage_in, usage_out, model_id, stream)


def make_handler(aggregator):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            pass

        def _send(self, status, content_type, payload):
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(payload)))
            self.send_header("connection", "close")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_body(self):
            length = self.headers.get("Content-Length")
            try:
                size = int(length) if length is not None else 0
            except ValueError:
                size = 0
            if size <= 0:
                return b""
            return self.rfile.read(min(size, 2 * 1024 * 1024))

        def do_GET(self):
            if self.path.split("?")[0] == "/v1/models":
                api = "anthropic" if self.headers.get("x-api-key") else "openai"
                payload = json.dumps(aggregator.models_list(api)).encode("utf-8")
                self._send(200, "application/json", payload)
                return
            self._send(404, "application/json", b'{"error": "not_found"}')

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/v1/chat/completions":
                status, content_type, payload = aggregator.handle_request("openai", self._read_body())
                self._send(status, content_type, payload)
            elif path == "/v1/messages":
                status, content_type, payload = aggregator.handle_request("anthropic", self._read_body())
                self._send(status, content_type, payload)
            else:
                self._send(404, "application/json", b'{"error": "not_found"}')

    return Handler


# --------------------------------------------------------------------------
# Startup: build the catalog, print it for --list, else register and serve
# until the env says the workload is done.
# --------------------------------------------------------------------------

def build_startup_catalog(env_url, providers, ratings):
    probes = [probe_provider(env_url, provider) for provider in providers]
    for provider, probe in zip(providers, probes):
        ok, status, models, err = probe
        log("probe %s: ok=%s status=%s models=%d %s" % (provider["id"], ok, status, len(models), err))
    return build_catalog(providers, probes, ratings)


def http_json(method, url, payload, timeout=10.0):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            return err.code, json.loads(err.read().decode("utf-8"))
        except Exception:
            return err.code, {}
    except Exception as err:
        return 0, {"error": str(err)}


def register_with_env(env_url, port, catalog):
    for attempt in range(5):
        status, answer = http_json(
            "POST", env_url.rstrip("/") + "/register",
            {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
        )
        if status == 200 and answer.get("ok") is True:
            log("registered on port %d, %s requests" % (port, answer.get("requests")))
            return True
        log("register attempt %d failed: HTTP %s %s" % (attempt + 1, status, answer))
        time.sleep(0.5)
    return False


def work_done(env_url):
    try:
        status, state = http_json("GET", env_url.rstrip("/") + "/work", None, timeout=WORK_TIMEOUT)
        return status == 200 and state.get("done") is True
    except Exception:
        return False


def main(argv):
    list_only = "--list" in argv
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    env_url = os.environ.get("MAKEFUN_ENV_URL", "")
    providers, ratings, manifest = [], {"ratings": [], "best": []}, []
    if prompt_file and os.path.exists(prompt_file):
        providers, ratings, manifest = load_brief(prompt_file)
    if list_only:
        if providers and env_url:
            catalog = build_startup_catalog(env_url, providers, ratings)
        else:
            catalog = EMPTY_CATALOG
        sys.stdout.write(catalog_text(catalog) + "\n")
        return 0
    if not env_url:
        log("MAKEFUN_ENV_URL is not set; nothing to serve")
        return 1
    if not providers:
        log("no provider directory in the brief; cannot build a catalog")
        return 1
    catalog = build_startup_catalog(env_url, providers, ratings)
    aggregator = Aggregator(env_url, catalog, ratings, manifest, providers)
    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(aggregator))
    server.daemon_threads = True
    port = server.server_address[1]
    if not register_with_env(env_url, port, catalog):
        log("registration failed; giving up")
        return 1
    stop = threading.Event()

    def on_signal(_signum, _frame):
        stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    log("serving on port %d" % port)
    while not stop.is_set():
        if work_done(env_url):
            break
        stop.wait(WORK_POLL_INTERVAL)
    server.shutdown()
    server.server_close()
    log("workload done; exiting 0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
