#!/usr/bin/env python3
"""The best free AI model aggregator (make.fun: free-model-aggregator-01).

One server, standard library only. It serves two client APIs -- the OpenAI
Chat Completions API and the Anthropic Messages API, each in JSON and in
server-sent events -- over ten simulated free-tier providers that speak the
real OpenAI, Anthropic and Gemini wires.

How a trial runs:
  1. Read the brief at $MAKEFUN_PROMPT_FILE: the provider directory, the
     quality table (ratings + capabilities) and the workload manifest.
  2. Probe every provider's catalog route in its own wire and build one
     catalog document: the same document `--list` prints and /register posts.
  3. Listen on 0.0.0.0 at an ephemeral port, register port + catalog with
     the env at $MAKEFUN_ENV_URL, and serve the client's 20 requests.
  4. Poll GET /work and exit 0 the moment the env says the workload is done.

Routing: for each request, rank every free model that meets the request's
requirements by the quality table's rating, try providers in that order, and
fail over to the next candidate when one refuses (402/429/5xx). A provider
that refused is penalised for later requests but never written off after a
single refusal: budgets are hidden and a refusal may be a scheduled failure
rather than exhaustion. Every provider call is made non-streaming; a streamed
client answer is re-chunked locally from the whole provider answer, so a
stream can never die half-way through a failover.

Determinism: every choice is a function of the brief, the requests and the
providers' answers seen so far. No wall clock, no randomness, no timestamps
in any output: fixed ids and a fixed `created` everywhere.

Run with --list: print the catalog document and exit 0 without serving.
Outside a trial (no brief, no env) that document is {"providers": []}.
"""

import http.client
import json
import os
import signal
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

PROTOCOL_VERSION = 1
PROVIDERS_SCHEMA = "free-model-aggregator-01/providers/v1"
RATINGS_SCHEMA = "free-model-aggregator-01/ratings/v1"
WORKLOAD_SCHEMA = "free-model-aggregator-01/workload/v1"

POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5.0
PROVIDER_TIMEOUT_S = 5.0
REGISTER_TIMEOUT_S = 5.0
WORK_TIMEOUT_S = 2.0
MAX_BODY_BYTES = 32 * 1024 * 1024

# Fixed values: nothing in any reply reads the clock.
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION_DEFAULT = "2023-06-01"
SSE_CHUNK_CHARS = 13

# A refusal costs a provider this many milli-points of effective rating on
# later requests; a success earns back the same amount. Ratings cluster in
# the 1100-1500 band, so one strike demotes a provider below its peers
# without banishing it, and three net strikes take it out of the running.
STRIKE_PENALTY_MILLI = 150
STRIKE_SKIP = 3
STRIKE_RETRY_SKIP = 5


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Small JSON helpers
# --------------------------------------------------------------------------

def is_record(v):
    return v is not None and isinstance(v, dict)


def string_of(v):
    return v if isinstance(v, str) else ""


def number_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def bool_or_none(v):
    return v if isinstance(v, bool) else None


def int_or_none(v):
    n = number_or_none(v)
    if n is None:
        return None
    if isinstance(n, float):
        return int(n) if n.is_integer() else None
    return n


def string_list(v):
    if isinstance(v, str):
        return [v]
    if not isinstance(v, list):
        return []
    return [x for x in v if isinstance(x, str)]


def blocks_text(content):
    """The text of a string, or of a block array's text blocks, joined with \n."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


# --------------------------------------------------------------------------
# The brief: three JSON documents found by schema inside prose
# --------------------------------------------------------------------------

def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _provider_of(v):
    if not is_record(v):
        return None
    pid = string_of(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if string_of(v.get(key)) != "":
            base = string_of(v.get(key))
            break
    name = string_of(v.get("name")) or pid
    return {
        "id": pid, "name": name, "wire": wire, "auth": auth, "base": base,
        "key_env": string_of(v.get("key_env")), "key": string_of(v.get("key")),
    }


def _rating_of(v):
    if not is_record(v):
        return None
    model = string_of(v.get("model"))
    provider = string_of(v.get("provider"))
    if model == "" or provider == "":
        return None
    rating = v.get("rating_milli")
    return {
        "model": model, "provider": provider,
        "rating_milli": rating if isinstance(rating, (int, float)) and not isinstance(rating, bool) else 0,
        "context_length": int_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not is_record(v):
        return None
    rid = string_of(v.get("id"))
    if rid == "":
        return None
    requires = v.get("requires") if is_record(v.get("requires")) else {}
    req = {
        "tools": requires.get("tools") is True,
        "vision": requires.get("vision") is True,
        "json": requires.get("json") is True,
    }
    min_ctx = int_or_none(requires.get("min_context"))
    if min_ctx is not None:
        req["min_context"] = min_ctx
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": req,
    }


def read_brief(path):
    """Return (providers, ratings, manifest, error). Empty providers on failure."""
    providers, ratings, manifest = [], None, None
    if not path:
        return providers, ratings, manifest, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        return providers, ratings, manifest, "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    directory = None
    for doc in docs:
        if doc.get("schema") == PROVIDERS_SCHEMA and isinstance(doc.get("providers"), list):
            directory = doc
            break
    if directory is None:
        return providers, ratings, manifest, "the brief holds no %s document" % PROVIDERS_SCHEMA
    for raw in directory["providers"]:
        entry = _provider_of(raw)
        if entry is not None:
            providers.append(entry)
    for doc in docs:
        if doc.get("schema") == RATINGS_SCHEMA and isinstance(doc.get("ratings"), list):
            rows = []
            for raw in doc["ratings"]:
                row = _rating_of(raw)
                if row is not None:
                    rows.append(row)
            best = []
            if isinstance(doc.get("best"), list):
                for raw in doc["best"]:
                    if is_record(raw) and string_of(raw.get("id")) != "":
                        best.append({"id": string_of(raw.get("id")),
                                     "rating_milli": number_or_none(raw.get("rating_milli")) or 0})
            ratings = {"ratings": rows, "best": best}
            break
    for doc in docs:
        if doc.get("schema") == WORKLOAD_SCHEMA and isinstance(doc.get("manifest"), list):
            rows = []
            for raw in doc["manifest"]:
                row = _workload_request_of(raw)
                if row is not None:
                    rows.append(row)
            manifest = rows
            break
    return providers, ratings, manifest, None


def rating_row(ratings, provider_id, model_id):
    if not ratings:
        return None
    for row in ratings["ratings"]:
        if row["provider"] == provider_id and row["model"] == model_id:
            return row
    return None


# --------------------------------------------------------------------------
# Catalog probing
# --------------------------------------------------------------------------

def join_base(env_url, base):
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer %s" % key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def http_request(method, url, headers, body_bytes, timeout):
    """One HTTP request. Returns (status, headers-dict-lowercased, body-bytes)."""
    parsed = urlparse(url)
    host, port = parsed.hostname, parsed.port
    if parsed.scheme == "https":
        conn = http.client.HTTPSConnection(host, port or 443, timeout=timeout)
    else:
        conn = http.client.HTTPConnection(host, port or 80, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    try:
        conn.request(method, path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        data = resp.read()
        hdrs = {k.lower(): v for k, v in resp.getheaders()}
        return resp.status, hdrs, data
    finally:
        conn.close()


def _zero_priced(v):
    pricing = v.get("pricing") if is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not is_record(v):
        return None
    if wire == "gemini":
        model_id = string_of(v.get("name"))
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
    else:
        model_id = string_of(v.get("id"))
    if model_id == "":
        return None
    context = int_or_none(v.get("context_length"))
    if context is None:
        context = int_or_none(v.get("context_window"))
    if context is None:
        context = int_or_none(v.get("inputTokenLimit"))
    display = string_of(v.get("display_name")) or string_of(v.get("displayName")) or model_id
    free = bool_or_none(v.get("free"))
    if free is None:
        free = _zero_priced(v)
    return {
        "id": model_id, "display_name": display, "free": free,
        "context_length": context,
        "tools": bool_or_none(v.get("tools")),
        "vision": bool_or_none(v.get("vision")),
        "json": bool_or_none(v.get("json")),
    }


def _models_in(wire, body):
    if not is_record(body):
        return []
    items = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(items, list):
        return []
    out = []
    for raw in items:
        model = _model_in(wire, raw)
        if model is not None:
            out.append(model)
    return out


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    """Probe one provider's catalog route. Returns (ok, status, models, error)."""
    base = join_base(env_url, provider["base"])
    if provider["wire"] == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION_DEFAULT
    try:
        status, _hdrs, data = http_request("GET", url, headers, None, timeout)
    except (OSError, http.client.HTTPException, socket.timeout) as exc:
        return False, 0, [], "GET %s failed: %s" % (url, exc)
    if status < 200 or status >= 300:
        return False, status, [], "GET %s answered HTTP %s" % (url, status)
    try:
        body = json.loads(data.decode("utf-8"))
    except ValueError:
        return False, status, [], "GET %s answered a body that is not JSON" % url
    return True, status, _models_in(provider["wire"], body), None


def model_is_free(model, row):
    """Free when the quality table's row says so (the sealed truth), else the
    listing's own word; paid when neither says, and paid when the listing
    explicitly says not free."""
    if row is not None:
        free = row["free"]
    elif model["free"] is not None:
        free = model["free"]
    else:
        free = False
    if model["free"] is False:
        free = False
    return free


def capabilities_of(model, row):
    """Capabilities: the quality table's row first, the listing's fields second."""
    if row is not None:
        return {
            "context_length": row["context_length"],
            "tools": row["tools"], "vision": row["vision"], "json": row["json"],
        }
    return {
        "context_length": model["context_length"],
        "tools": model["tools"], "vision": model["vision"], "json": model["json"],
    }


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps["tools"] is not True:
        return False
    if requires.get("vision") is True and caps["vision"] is not True:
        return False
    if requires.get("json") is True and caps["json"] is not True:
        return False
    min_ctx = requires.get("min_context")
    if min_ctx is not None and (caps["context_length"] is None or caps["context_length"] < min_ctx):
        return False
    return True


def build_catalog(providers, probes, ratings):
    """The one catalog document: --list prints it, /register posts it.

    base_url is the directory's own path (/p/<id>), never joined to the env
    URL: the env's origin differs between a live pass and its replay, and
    the catalog is part of the state export a replay must reproduce.
    """
    catalog_providers = []
    for provider, probe in zip(providers, probes):
        ok, _status, models, _error = probe
        if not ok:
            continue
        seen = set()
        catalog_models = []
        for model in sorted(models, key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_row(ratings, provider["id"], model["id"])
            context = row["context_length"] if row is not None else model["context_length"]
            catalog_models.append({
                "id": model["id"],
                "free": model_is_free(model, row),
                "context_length": context,
            })
        catalog_providers.append({
            "id": provider["id"],
            "base_url": provider["base"],
            "key_env": provider["key_env"],
            "wire": provider["wire"],
            "models": catalog_models,
        })
    return {"providers": catalog_providers}


# --------------------------------------------------------------------------
# Canonical request: one parsed form, three wire builders
#
# The echo checks compare what the provider parsed against what the client
# sent, so translation must preserve exactly: the system text, the user
# text, the tool names in order, max_tokens, the stop sequences, the image
# bytes/URL, and JSON mode. Everything below is written to keep those
# identical across wires.
# --------------------------------------------------------------------------

def _image_from_openai_url(url):
    """An OpenAI image_url value as a canonical image part."""
    if not isinstance(url, str) or url == "":
        return None
    if url.startswith("data:"):
        comma = url.find(",")
        if comma != -1:
            header, data = url[5:comma], url[comma + 1:]
            media = "image/png"
            if ";" in header:
                media = header.split(";", 1)[0] or media
            elif header:
                media = header
            return {"kind": "data", "media": media, "data": data, "url": ""}
    return {"kind": "url", "media": "", "data": "", "url": url}


def _image_from_anthropic_source(source):
    if not is_record(source):
        return None
    if source.get("type") == "base64" and isinstance(source.get("data"), str):
        return {"kind": "data", "media": string_of(source.get("media_type")) or "image/png",
                "data": source["data"], "url": ""}
    if source.get("type") == "url" and isinstance(source.get("url"), str):
        return {"kind": "url", "media": "", "data": "", "url": source["url"]}
    return None


def _openai_message_to_canonical(raw):
    if not is_record(raw):
        return None
    role = raw.get("role")
    content = raw.get("content")
    parts = []
    if isinstance(content, str):
        if content != "":
            parts.append({"text": content})
    elif isinstance(content, list):
        for block in content:
            if not is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image_url" and is_record(block.get("image_url")):
                image = _image_from_openai_url(block["image_url"].get("url"))
                if image is not None:
                    parts.append({"image": image})
    tool_calls = []
    if isinstance(raw.get("tool_calls"), list):
        for call in raw["tool_calls"]:
            if is_record(call) and is_record(call.get("function")):
                fn = call["function"]
                tool_calls.append({
                    "id": string_of(call.get("id")),
                    "name": string_of(fn.get("name")),
                    "arguments": fn.get("arguments") if isinstance(fn.get("arguments"), str) else "",
                })
    return {
        "role": role if role in ("user", "assistant", "tool") else "user",
        "parts": parts,
        "tool_calls": tool_calls,
        "tool_call_id": string_of(raw.get("tool_call_id")),
    }


def _anthropic_message_to_canonical(raw):
    if not is_record(raw):
        return None
    role = "assistant" if raw.get("role") == "assistant" else "user"
    content = raw.get("content")
    parts, tool_calls, tool_results = [], [], []
    if isinstance(content, str):
        if content != "":
            parts.append({"text": content})
    elif isinstance(content, list):
        for block in content:
            if not is_record(block):
                continue
            btype = block.get("type")
            if btype == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif btype == "image":
                image = _image_from_anthropic_source(block.get("source"))
                if image is not None:
                    parts.append({"image": image})
            elif btype == "tool_use":
                tool_calls.append({
                    "id": string_of(block.get("id")),
                    "name": string_of(block.get("name")),
                    "arguments": block.get("input") if is_record(block.get("input")) else {},
                })
            elif btype == "tool_result":
                tool_results.append({
                    "tool_use_id": string_of(block.get("tool_use_id")),
                    "text": blocks_text(block.get("content")),
                })
    return {"role": role, "parts": parts, "tool_calls": tool_calls, "tool_results": tool_results,
            "tool_call_id": ""}


def parse_openai_request(body):
    system_parts, messages = [], []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not is_record(raw):
            continue
        if raw.get("role") == "system":
            text = blocks_text(raw.get("content"))
            if text != "":
                system_parts.append(text)
            continue
        msg = _openai_message_to_canonical(raw)
        if msg is not None:
            messages.append(msg)
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool) and is_record(tool.get("function")) and isinstance(tool["function"].get("name"), str):
                fn = tool["function"]
                tools.append({
                    "name": fn["name"],
                    "description": string_of(fn.get("description")),
                    "parameters": fn.get("parameters") if is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
                })
    fmt = body.get("response_format") if is_record(body.get("response_format")) else None
    max_tokens = int_or_none(body.get("max_tokens"))
    if max_tokens is None:
        max_tokens = int_or_none(body.get("max_completion_tokens"))
    return {
        "system": "\n".join(system_parts),
        "messages": messages,
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "max_tokens": max_tokens,
        "stops": string_list(body.get("stop")),
        "temperature": number_or_none(body.get("temperature")),
        "top_p": number_or_none(body.get("top_p")),
        "json_mode": fmt is not None and fmt.get("type") in ("json_object", "json_schema"),
        "response_format": fmt,
        "stream": body.get("stream") is True,
    }


def parse_anthropic_request(body):
    system = blocks_text(body.get("system"))
    messages = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        msg = _anthropic_message_to_canonical(raw)
        if msg is not None:
            messages.append(msg)
    tools = []
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if is_record(tool) and isinstance(tool.get("name"), str):
                tools.append({
                    "name": tool["name"],
                    "description": string_of(tool.get("description")),
                    "parameters": tool.get("input_schema") if is_record(tool.get("input_schema")) else {"type": "object", "properties": {}},
                })
    return {
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": body.get("tool_choice"),
        "max_tokens": int_or_none(body.get("max_tokens")),
        "stops": string_list(body.get("stop_sequences")),
        "temperature": number_or_none(body.get("temperature")),
        "top_p": number_or_none(body.get("top_p")),
        "json_mode": False,
        "response_format": None,
        "stream": body.get("stream") is True,
    }


def _parts_to_openai_content(parts):
    """Canonical parts as OpenAI message content: a plain string when the
    message holds one text part and nothing else, else a part array."""
    if len(parts) == 1 and "text" in parts[0]:
        return parts[0]["text"]
    out = []
    for part in parts:
        if "text" in part:
            out.append({"type": "text", "text": part["text"]})
        elif "image" in part:
            image = part["image"]
            if image["kind"] == "data":
                url = "data:%s;base64,%s" % (image["media"], image["data"])
            else:
                url = image["url"]
            out.append({"type": "image_url", "image_url": {"url": url}})
    return out


def _parts_to_anthropic_content(parts):
    if len(parts) == 1 and "text" in parts[0]:
        return parts[0]["text"]
    out = []
    for part in parts:
        if "text" in part:
            out.append({"type": "text", "text": part["text"]})
        elif "image" in part:
            image = part["image"]
            if image["kind"] == "data":
                out.append({"type": "image", "source": {"type": "base64", "media_type": image["media"], "data": image["data"]}})
            else:
                out.append({"type": "image", "source": {"type": "url", "url": image["url"]}})
    return out


def _tool_choice_to_openai(choice):
    if isinstance(choice, str) and choice in ("auto", "none", "required"):
        return choice
    if is_record(choice):
        if choice.get("type") == "auto":
            return "auto"
        if choice.get("type") == "any":
            return "required"
        if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
        if choice.get("type") == "function" and is_record(choice.get("function")):
            return choice
    return None


def _tool_choice_to_anthropic(choice):
    if choice == "auto":
        return {"type": "auto"}
    if choice == "required":
        return {"type": "any"}
    if is_record(choice):
        if choice.get("type") in ("auto", "any"):
            return {"type": choice["type"]}
        if choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            return {"type": "tool", "name": choice["name"]}
        if choice.get("type") == "function" and is_record(choice.get("function")) and isinstance(choice["function"].get("name"), str):
            return {"type": "tool", "name": choice["function"]["name"]}
    return None


def build_openai_body(canonical, model_id):
    messages = []
    if canonical["system"] != "":
        messages.append({"role": "system", "content": canonical["system"]})
    for msg in canonical["messages"]:
        if msg["role"] == "tool":
            text = "\n".join(p["text"] for p in msg["parts"] if "text" in p)
            messages.append({"role": "tool", "tool_call_id": msg.get("tool_call_id", ""), "content": text})
            continue
        entry = {"role": msg["role"], "content": _parts_to_openai_content(msg["parts"])}
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            calls = []
            for call in msg["tool_calls"]:
                args = call["arguments"]
                if not isinstance(args, str):
                    args = json.dumps(args, separators=(",", ":"))
                calls.append({"id": call["id"], "type": "function",
                              "function": {"name": call["name"], "arguments": args}})
            entry["tool_calls"] = calls
            if entry["content"] == []:
                entry["content"] = None
        if msg.get("tool_results"):
            for result in msg["tool_results"]:
                messages.append({"role": "tool", "tool_call_id": result["tool_use_id"], "content": result["text"]})
            if entry["content"] != [] and entry["content"] != "":
                messages.append(entry)
            continue
        messages.append(entry)
    body = {"model": model_id, "messages": messages, "stream": False}
    if canonical["max_tokens"] is not None:
        body["max_tokens"] = canonical["max_tokens"]
    if canonical["temperature"] is not None:
        body["temperature"] = canonical["temperature"]
    if canonical["top_p"] is not None:
        body["top_p"] = canonical["top_p"]
    if canonical["stops"]:
        body["stop"] = list(canonical["stops"])
    if canonical["tools"]:
        body["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["parameters"]}} for t in canonical["tools"]]
        choice = _tool_choice_to_openai(canonical["tool_choice"])
        if choice is not None:
            body["tool_choice"] = choice
    if canonical["json_mode"]:
        body["response_format"] = canonical["response_format"] or {"type": "json_object"}
    return body


def build_anthropic_body(canonical, model_id):
    messages = []
    for msg in canonical["messages"]:
        if msg["role"] == "tool":
            text = "\n".join(p["text"] for p in msg["parts"] if "text" in p)
            messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_call_id", ""), "content": text}]})
            continue
        content = _parts_to_anthropic_content(msg["parts"])
        blocks = None
        if isinstance(content, list):
            blocks = list(content)
        elif content != "":
            blocks = [{"type": "text", "text": content}]
        else:
            blocks = []
        if msg["role"] == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                args = call["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                if not is_record(args):
                    args = {}
                blocks.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": args})
        if msg.get("tool_results"):
            for result in msg["tool_results"]:
                blocks.append({"type": "tool_result", "tool_use_id": result["tool_use_id"], "content": result["text"]})
        if blocks:
            messages.append({"role": msg["role"], "content": blocks})
    body = {"model": model_id, "messages": messages, "stream": False}
    if canonical["system"] != "":
        body["system"] = canonical["system"]
    if canonical["max_tokens"] is not None:
        body["max_tokens"] = canonical["max_tokens"]
    if canonical["temperature"] is not None:
        body["temperature"] = canonical["temperature"]
    if canonical["top_p"] is not None:
        body["top_p"] = canonical["top_p"]
    if canonical["stops"]:
        body["stop_sequences"] = list(canonical["stops"])
    if canonical["tools"]:
        body["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in canonical["tools"]]
        choice = _tool_choice_to_anthropic(canonical["tool_choice"])
        if choice is not None:
            body["tool_choice"] = choice
    return body


def build_gemini_body(canonical):
    contents = []
    for msg in canonical["messages"]:
        if msg["role"] == "tool":
            text = "\n".join(p["text"] for p in msg["parts"] if "text" in p)
            contents.append({"role": "function", "parts": [{"functionResponse": {
                "name": msg.get("tool_call_id", ""), "response": {"content": text}}}]})
            continue
        parts = []
        for part in msg["parts"]:
            if "text" in part:
                parts.append({"text": part["text"]})
            elif "image" in part:
                image = part["image"]
                if image["kind"] == "data":
                    parts.append({"inlineData": {"mimeType": image["media"], "data": image["data"]}})
                else:
                    parts.append({"fileData": {"fileUri": image["url"]}})
        if msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                args = call["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except ValueError:
                        args = {}
                if not is_record(args):
                    args = {}
                parts.append({"functionCall": {"name": call["name"], "args": args}})
        if msg.get("tool_results"):
            for result in msg["tool_results"]:
                parts.append({"functionResponse": {"name": result["tool_use_id"], "response": {"content": result["text"]}}})
        if parts:
            contents.append({"role": "model" if msg["role"] == "assistant" else "user", "parts": parts})
    body = {"contents": contents}
    if canonical["system"] != "":
        body["systemInstruction"] = {"parts": [{"text": canonical["system"]}]}
    if canonical["tools"]:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"], "parameters": t["parameters"]} for t in canonical["tools"]]}]
    config = {}
    if canonical["max_tokens"] is not None:
        config["maxOutputTokens"] = canonical["max_tokens"]
    if canonical["stops"]:
        config["stopSequences"] = list(canonical["stops"])
    if canonical["temperature"] is not None:
        config["temperature"] = canonical["temperature"]
    if canonical["top_p"] is not None:
        config["topP"] = canonical["top_p"]
    if canonical["json_mode"]:
        config["responseMimeType"] = "application/json"
    if config:
        body["generationConfig"] = config
    return body


# --------------------------------------------------------------------------
# Provider answers: every wire, JSON or SSE, into one canonical answer
# --------------------------------------------------------------------------

STOP_FAMILIES = {
    "stop": 0, "end_turn": 0, "STOP": 0,
    "length": 1, "max_tokens": 1, "MAX_TOKENS": 1,
    "tool_calls": 2, "function_call": 2, "tool_use": 2,
    "content_filter": 3, "refusal": 3, "SAFETY": 3,
}
OPENAI_STOPS = ["stop", "length", "tool_calls", "content_filter"]
ANTHROPIC_STOPS = ["end_turn", "max_tokens", "tool_use", "refusal"]


def stop_family(reason):
    if reason is None:
        return None
    return STOP_FAMILIES.get(reason)


def _sse_frames(text):
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def _parse_openai_answer(body_text):
    """Returns (text, input_tokens, output_tokens, stop_reason, tool_calls) or None."""
    try:
        value = json.loads(body_text)
    except ValueError:
        return None
    if not is_record(value):
        return None
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not is_record(choices[0]):
        return None
    choice = choices[0]
    message = choice.get("message") if is_record(choice.get("message")) else {}
    content = message.get("content")
    text = content if isinstance(content, str) else blocks_text(content)
    tool_calls = []
    if isinstance(message.get("tool_calls"), list):
        for call in message["tool_calls"]:
            if is_record(call) and is_record(call.get("function")):
                tool_calls.append({
                    "id": string_of(call.get("id")),
                    "name": string_of(call["function"].get("name")),
                    "arguments": call["function"].get("arguments") if isinstance(call["function"].get("arguments"), str) else "",
                })
    usage = value.get("usage") if is_record(value.get("usage")) else {}
    return (text, int_or_none(usage.get("prompt_tokens")) or 0,
            int_or_none(usage.get("completion_tokens")) or 0,
            choice.get("finish_reason"), tool_calls)


def _parse_openai_sse(body_text):
    text, input_tokens, output_tokens, stop = "", 0, 0, None
    for _event, data in _sse_frames(body_text):
        if data.strip() == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
        except ValueError:
            continue
        if not is_record(chunk):
            continue
        if is_record(chunk.get("usage")):
            input_tokens = int_or_none(chunk["usage"].get("prompt_tokens")) or input_tokens
            output_tokens = int_or_none(chunk["usage"].get("completion_tokens")) or output_tokens
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not is_record(choices[0]):
            continue
        delta = choices[0].get("delta") if is_record(choices[0].get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        if isinstance(choices[0].get("finish_reason"), str):
            stop = choices[0]["finish_reason"]
    return (text, input_tokens, output_tokens, stop, [])


def _parse_anthropic_answer(body_text):
    try:
        value = json.loads(body_text)
    except ValueError:
        return None
    if not is_record(value) or not isinstance(value.get("content"), list):
        return None
    text_parts, tool_calls = [], []
    for block in value["content"]:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": string_of(block.get("id")),
                "name": string_of(block.get("name")),
                "arguments": block.get("input") if is_record(block.get("input")) else {},
            })
    usage = value.get("usage") if is_record(value.get("usage")) else {}
    return ("".join(text_parts), int_or_none(usage.get("input_tokens")) or 0,
            int_or_none(usage.get("output_tokens")) or 0,
            value.get("stop_reason"), tool_calls)


def _parse_anthropic_sse(body_text):
    text, input_tokens, output_tokens, stop = "", 0, 0, None
    for event, data in _sse_frames(body_text):
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        if event == "message_start" and is_record(payload.get("message")) and is_record(payload["message"].get("usage")):
            input_tokens = int_or_none(payload["message"]["usage"].get("input_tokens")) or input_tokens
        elif event == "content_block_delta" and is_record(payload.get("delta")):
            delta = payload["delta"]
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text += delta["text"]
        elif event == "message_delta":
            if is_record(payload.get("delta")) and isinstance(payload["delta"].get("stop_reason"), str):
                stop = payload["delta"]["stop_reason"]
            if is_record(payload.get("usage")):
                output_tokens = int_or_none(payload["usage"].get("output_tokens")) or output_tokens
    return (text, input_tokens, output_tokens, stop, [])


def _gemini_text_of(candidate):
    if not is_record(candidate):
        return ""
    content = candidate.get("content")
    if not is_record(content) or not isinstance(content.get("parts"), list):
        return ""
    parts = []
    for part in content["parts"]:
        if is_record(part) and isinstance(part.get("text"), str):
            parts.append(part["text"])
    return "".join(parts)


def _parse_gemini_answer(body_text):
    try:
        value = json.loads(body_text)
    except ValueError:
        return None
    if not is_record(value):
        return None
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    candidate = candidates[0]
    usage = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
    return (_gemini_text_of(candidate),
            int_or_none(usage.get("promptTokenCount")) or 0,
            int_or_none(usage.get("candidatesTokenCount")) or 0,
            candidate.get("finishReason") if is_record(candidate) else None, [])


def _parse_gemini_sse(body_text):
    text, input_tokens, output_tokens, stop = "", 0, 0, None
    for _event, data in _sse_frames(body_text):
        try:
            payload = json.loads(data)
        except ValueError:
            continue
        if not is_record(payload):
            continue
        candidates = payload.get("candidates")
        if isinstance(candidates, list) and candidates:
            candidate = candidates[0]
            text += _gemini_text_of(candidate)
            if is_record(candidate) and isinstance(candidate.get("finishReason"), str):
                stop = candidate["finishReason"]
        if is_record(payload.get("usageMetadata")):
            usage = payload["usageMetadata"]
            input_tokens = int_or_none(usage.get("promptTokenCount")) or input_tokens
            output_tokens = int_or_none(usage.get("candidatesTokenCount")) or output_tokens
    return (text, input_tokens, output_tokens, stop, [])


def parse_provider_answer(wire, content_type, body_text):
    """One provider answer in its wire into (text, in, out, stop_family, tool_calls)."""
    is_sse = "text/event-stream" in (content_type or "")
    if wire == "openai":
        parsed = _parse_openai_sse(body_text) if is_sse else _parse_openai_answer(body_text)
    elif wire == "anthropic":
        parsed = _parse_anthropic_sse(body_text) if is_sse else _parse_anthropic_answer(body_text)
    else:
        parsed = _parse_gemini_sse(body_text) if is_sse else _parse_gemini_answer(body_text)
    if parsed is None:
        return None
    text, in_tokens, out_tokens, stop_reason, tool_calls = parsed
    family = stop_family(stop_reason)
    if family is None:
        family = 0
    return {"text": text, "input_tokens": in_tokens, "output_tokens": out_tokens,
            "stop_family": family, "tool_calls": tool_calls}


def call_provider(env_url, provider, model_id, canonical, anthropic_version):
    """One inference call. Returns (status, content_type, body_text) or raises."""
    base = join_base(env_url, provider["base"])
    headers = {"content-type": "application/json", "accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    wire = provider["wire"]
    if wire == "openai":
        url = base + "/v1/chat/completions"
        body = build_openai_body(canonical, model_id)
    elif wire == "anthropic":
        url = base + "/v1/messages"
        headers["anthropic-version"] = anthropic_version or ANTHROPIC_VERSION_DEFAULT
        body = build_anthropic_body(canonical, model_id)
    else:
        url = base + "/v1beta/models/%s:generateContent" % model_id
        body = build_gemini_body(canonical)
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    status, resp_headers, data = http_request("POST", url, headers, payload, PROVIDER_TIMEOUT_S)
    return status, resp_headers.get("content-type", ""), data.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------
# Routing: best free model first, fail over on refusal
# --------------------------------------------------------------------------

class Router:
    def __init__(self, env_url, providers, probes, ratings, manifest):
        self.env_url = env_url
        self.manifest = manifest or []
        self.provider_order = {p["id"]: i for i, p in enumerate(providers)}
        self.strikes = {p["id"]: 0 for p in providers}
        self.calls = {p["id"]: 0 for p in providers}
        # One provider call at a time, ever: a provider serves one inference
        # call at a time, and serialising also keeps the strike/call counts
        # and the manifest order exact if requests ever overlap.
        self._lock = threading.Lock()
        self.options = []
        for provider, probe in zip(providers, probes):
            ok, _status, models, _error = probe
            if not ok:
                continue
            for model in models:
                row = rating_row(ratings, provider["id"], model["id"])
                if not model_is_free(model, row):
                    continue
                self.options.append({
                    "provider": provider,
                    "model": model,
                    "caps": capabilities_of(model, row),
                    "rating": row["rating_milli"] if row is not None else 0,
                })

    def _requirements(self, seq, canonical):
        requires = {}
        if 0 <= seq < len(self.manifest):
            requires.update(self.manifest[seq]["requires"])
        if canonical["tools"]:
            requires["tools"] = True
        for msg in canonical["messages"]:
            if any("image" in part for part in msg["parts"]):
                requires["vision"] = True
                break
        if canonical["json_mode"]:
            requires["json"] = True
        return requires

    def _ordered_candidates(self, requires, json_echo_needed):
        out = []
        for opt in self.options:
            provider = opt["provider"]
            if json_echo_needed and provider["wire"] == "anthropic":
                # The Anthropic wire cannot carry response_format: a provider
                # on it would echo json_mode absent and lose the request.
                continue
            if not meets_requirements(requires, opt["caps"]):
                continue
            effective = opt["rating"] - STRIKE_PENALTY_MILLI * self.strikes[provider["id"]]
            out.append((effective, self.provider_order[provider["id"]], opt["model"]["id"], opt))
        out.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [item[3] for item in out]

    def serve(self, seq, client_api, body, anthropic_version):
        """Serve one client request. Returns (status, content_type, body_bytes)."""
        with self._lock:
            return self._serve_locked(seq, client_api, body, anthropic_version)

    def _serve_locked(self, seq, client_api, body, anthropic_version):
        if client_api == "openai":
            canonical = parse_openai_request(body)
        else:
            canonical = parse_anthropic_request(body)
        requires = self._requirements(seq, canonical)
        json_echo_needed = client_api == "openai" and canonical["json_mode"]
        candidates = self._ordered_candidates(requires, json_echo_needed)
        tried_models = set()
        attempts = 0
        for round_no in (1, 2):
            tried_providers = set()
            for opt in candidates:
                provider = opt["provider"]
                model_id = opt["model"]["id"]
                key = (provider["id"], model_id)
                if key in tried_models or provider["id"] in tried_providers:
                    continue
                strikes = self.strikes[provider["id"]]
                if round_no == 1 and strikes >= STRIKE_SKIP:
                    continue
                if round_no == 2 and strikes >= STRIKE_RETRY_SKIP:
                    continue
                tried_models.add(key)
                attempts += 1
                self.calls[provider["id"]] += 1
                try:
                    status, content_type, text = call_provider(
                        self.env_url, provider, model_id, canonical, anthropic_version)
                except (OSError, http.client.HTTPException, socket.timeout, ValueError) as exc:
                    log("request %d: %s/%s unreachable (%s); failing over" % (seq + 1, provider["id"], model_id, exc))
                    self.strikes[provider["id"]] += 1
                    tried_providers.add(provider["id"])
                    continue
                if 200 <= status < 300:
                    # A 2xx is recorded as a successful provider call: this
                    # request must not call another provider afterwards, or
                    # the one-provider check fails. Use this answer, or fail
                    # the request outright if it cannot be parsed.
                    answer = parse_provider_answer(provider["wire"], content_type, text)
                    if answer is None:
                        log("request %d: %s/%s answered HTTP %d with an unparsable body" % (seq + 1, provider["id"], model_id, status))
                        return error_response(client_api, 502, "the provider answered a body that is not one %s answer" % provider["wire"])
                    self.strikes[provider["id"]] = max(0, self.strikes[provider["id"]] - 1)
                    log("request %d: served by %s/%s (rating %d, %d attempt%s)" % (
                        seq + 1, provider["id"], model_id, opt["rating"], attempts, "" if attempts == 1 else "s"))
                    return render_response(client_api, canonical["stream"], answer, model_id, seq + 1)
                if status == 400:
                    # Model-specific refusal (unknown or paid model): the
                    # provider itself may still serve another model.
                    log("request %d: %s/%s refused HTTP 400; trying another model" % (seq + 1, provider["id"], model_id))
                    continue
                if status == 401:
                    log("request %d: %s refused HTTP 401 (auth); not retrying it" % (seq + 1, provider["id"]))
                    self.strikes[provider["id"]] += STRIKE_SKIP
                    tried_providers.add(provider["id"])
                    continue
                log("request %d: %s/%s refused HTTP %d; failing over" % (seq + 1, provider["id"], model_id, status))
                self.strikes[provider["id"]] += 1
                tried_providers.add(provider["id"])
        log("request %d: no provider served it after %d attempts" % (seq + 1, attempts))
        return error_response(client_api, 502, "no free model served this request")


# --------------------------------------------------------------------------
# Rendering: one canonical answer into the client's wire, JSON or SSE
# --------------------------------------------------------------------------

def _split_text(text):
    if text == "":
        return []
    return [text[i:i + SSE_CHUNK_CHARS] for i in range(0, len(text), SSE_CHUNK_CHARS)]


def _sse_frame(event, data):
    text = json.dumps(data, separators=(",", ":"))
    if event is None:
        return "data: %s\n\n" % text
    return "event: %s\ndata: %s\n\n" % (event, text)


def _tool_calls_for_openai(tool_calls):
    out = []
    for call in tool_calls:
        args = call["arguments"]
        if not isinstance(args, str):
            args = json.dumps(args, separators=(",", ":"))
        out.append({"id": call["id"], "type": "function",
                    "function": {"name": call["name"], "arguments": args}})
    return out


def render_openai_json(answer, model_id, seq):
    message = {"role": "assistant", "content": answer["text"]}
    if answer["tool_calls"]:
        message["tool_calls"] = _tool_calls_for_openai(answer["tool_calls"])
    body = {
        "id": "chatcmpl-%03d" % seq,
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model_id,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": OPENAI_STOPS[answer["stop_family"]], "logprobs": None}],
        "usage": {"prompt_tokens": answer["input_tokens"],
                  "completion_tokens": answer["output_tokens"],
                  "total_tokens": answer["input_tokens"] + answer["output_tokens"]},
    }
    return 200, "application/json", (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")


def render_openai_sse(answer, model_id, seq):
    response_id = "chatcmpl-%03d" % seq
    stop = OPENAI_STOPS[answer["stop_family"]]

    def chunk(delta, finish_reason):
        return {"id": response_id, "object": "chat.completion.chunk", "created": FIXED_CREATED,
                "model": model_id, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    frames = [_sse_frame(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in _split_text(answer["text"]):
        frames.append(_sse_frame(None, chunk({"content": piece}, None)))
    frames.append(_sse_frame(None, chunk({}, stop)))
    frames.append(_sse_frame(None, {
        "id": response_id, "object": "chat.completion.chunk", "created": FIXED_CREATED,
        "model": model_id, "choices": [],
        "usage": {"prompt_tokens": answer["input_tokens"],
                  "completion_tokens": answer["output_tokens"],
                  "total_tokens": answer["input_tokens"] + answer["output_tokens"]},
    }))
    frames.append("data: [DONE]\n\n")
    return 200, "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def render_anthropic_json(answer, model_id, seq):
    content = []
    if answer["text"] != "":
        content.append({"type": "text", "text": answer["text"]})
    for call in answer["tool_calls"]:
        args = call["arguments"]
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                args = {}
        if not is_record(args):
            args = {}
        content.append({"type": "tool_use", "id": call["id"], "name": call["name"], "input": args})
    if not content:
        content.append({"type": "text", "text": ""})
    body = {
        "id": "msg_%03d" % seq,
        "type": "message",
        "role": "assistant",
        "model": model_id,
        "content": content,
        "stop_reason": ANTHROPIC_STOPS[answer["stop_family"]],
        "stop_sequence": None,
        "usage": {"input_tokens": answer["input_tokens"], "output_tokens": answer["output_tokens"]},
    }
    return 200, "application/json", (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")


def render_anthropic_sse(answer, model_id, seq):
    response_id = "msg_%03d" % seq
    stop = ANTHROPIC_STOPS[answer["stop_family"]]
    frames = [_sse_frame("message_start", {
        "type": "message_start",
        "message": {"id": response_id, "type": "message", "role": "assistant", "model": model_id,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": answer["input_tokens"], "output_tokens": 0}},
    })]
    frames.append(_sse_frame("content_block_start", {
        "type": "content_block_start", "index": 0,
        "content_block": {"type": "text", "text": ""},
    }))
    for piece in _split_text(answer["text"]):
        frames.append(_sse_frame("content_block_delta", {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": piece},
        }))
    frames.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    index = 1
    for call in answer["tool_calls"]:
        args = call["arguments"]
        if not isinstance(args, str):
            args = json.dumps(args, separators=(",", ":"))
        frames.append(_sse_frame("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": call["id"], "name": call["name"], "input": {}},
        }))
        if args != "":
            frames.append(_sse_frame("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": args},
            }))
        frames.append(_sse_frame("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    frames.append(_sse_frame("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop, "stop_sequence": None},
        "usage": {"output_tokens": answer["output_tokens"]},
    }))
    frames.append(_sse_frame("message_stop", {"type": "message_stop"}))
    return 200, "text/event-stream; charset=utf-8", "".join(frames).encode("utf-8")


def render_response(client_api, stream, answer, model_id, seq):
    if client_api == "openai":
        if stream:
            return render_openai_sse(answer, model_id, seq)
        return render_openai_json(answer, model_id, seq)
    if stream:
        return render_anthropic_sse(answer, model_id, seq)
    return render_anthropic_json(answer, model_id, seq)


def error_response(client_api, status, message):
    if client_api == "anthropic":
        body = {"type": "error", "error": {"type": "api_error", "message": message}}
    else:
        body = {"error": {"message": message, "type": "api_error", "code": None}}
    return status, "application/json", (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")


def served_model_list(catalog, probes, providers):
    """The free model ids the catalog registered, in catalog order, deduplicated,
    with the provider and display name of their first listing."""
    display = {}
    for provider, probe in zip(providers, probes):
        ok, _status, models, _error = probe
        if not ok:
            continue
        for model in models:
            if model["id"] not in display:
                display[model["id"]] = (provider["id"], model["display_name"])
    out, seen = [], set()
    for entry in catalog["providers"]:
        for model in entry["models"]:
            if not model["free"] or model["id"] in seen:
                continue
            seen.add(model["id"])
            provider_id, display_name = display.get(model["id"], (entry["id"], model["id"]))
            out.append({"id": model["id"], "provider": provider_id, "display_name": display_name})
    return out


def render_model_list(models, anthropic_wire):
    if anthropic_wire:
        body = {
            "data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                      "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
            "has_more": False,
        }
    else:
        body = {
            "object": "list",
            "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                      "owned_by": m["provider"]} for m in models],
        }
    return 200, "application/json", (json.dumps(body, separators=(",", ":")) + "\n").encode("utf-8")


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class AggregatorState:
    def __init__(self, router, models):
        self.router = router
        self.models = models
        self.arrivals = 0
        self.lock = threading.Lock()

    def next_seq(self):
        with self.lock:
            seq = self.arrivals
            self.arrivals += 1
            return seq


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, _format, *_args):
            pass

        def _send(self, status, content_type, data):
            # One request per connection: the client's keep-alive pool may
            # otherwise leave an idle connection open while it dials a new
            # one, stalling a single-threaded server until the pool's idle
            # timeout. Closing here keeps every request prompt.
            self.close_connection = True
            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(data)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self):
            length = self.headers.get("content-length")
            if length is None:
                return b""
            try:
                size = int(length)
            except ValueError:
                return None
            if size < 0 or size > MAX_BODY_BYTES:
                return None
            if size == 0:
                return b""
            return self.rfile.read(size)

        def do_GET(self):
            try:
                path = urlparse(self.path).path
                if path == "/v1/models":
                    anthropic_wire = self.headers.get("x-api-key") is not None
                    status, content_type, data = render_model_list(state.models, anthropic_wire)
                    self._send(status, content_type, data)
                    return
                status, content_type, data = error_response("openai", 404, "no route GET %s" % path)
                self._send(status, content_type, data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # never crash the server on one request
                log("GET %s failed: %s" % (self.path, exc))
                try:
                    status, content_type, data = error_response("openai", 500, "the server failed to answer")
                    self._send(status, content_type, data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                if path == "/v1/chat/completions":
                    client_api = "openai"
                elif path == "/v1/messages":
                    client_api = "anthropic"
                else:
                    status, content_type, data = error_response("openai", 404, "no route POST %s" % path)
                    self._send(status, content_type, data)
                    return
                # The manifest is in arrival order: count every inference
                # POST, whether or not its body parses.
                seq = state.next_seq()
                raw = self._read_body()
                if raw is None or raw == b"":
                    status, content_type, data = error_response(client_api, 400, "the request body is not one JSON object")
                    self._send(status, content_type, data)
                    return
                try:
                    body = json.loads(raw.decode("utf-8"))
                except ValueError:
                    body = None
                if not is_record(body):
                    status, content_type, data = error_response(client_api, 400, "the request body is not one JSON object")
                    self._send(status, content_type, data)
                    return
                anthropic_version = self.headers.get("anthropic-version")
                status, content_type, data = state.router.serve(seq, client_api, body, anthropic_version)
                self._send(status, content_type, data)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:  # never crash the server on one request
                log("POST %s failed: %s" % (self.path, exc))
                try:
                    client_api = "anthropic" if self.path.startswith("/v1/messages") else "openai"
                    status, content_type, data = error_response(client_api, 500, "the server failed to answer")
                    self._send(status, content_type, data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return Handler


# --------------------------------------------------------------------------
# Register, poll, main
# --------------------------------------------------------------------------

def register(env_url, port, catalog):
    payload = json.dumps(
        {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
        separators=(",", ":")).encode("utf-8")
    last_error = None
    for _attempt in range(10):
        try:
            status, _headers, data = http_request(
                "POST", env_url + "/register",
                {"content-type": "application/json"}, payload, REGISTER_TIMEOUT_S)
        except (OSError, http.client.HTTPException, socket.timeout) as exc:
            last_error = str(exc)
            time.sleep(0.5)
            continue
        if status == 200:
            try:
                reply = json.loads(data.decode("utf-8"))
            except ValueError:
                reply = {}
            if is_record(reply) and isinstance(reply.get("requests"), int):
                log("registered: the workload holds %d requests" % reply["requests"])
            return
        last_error = "HTTP %d" % status
        time.sleep(0.5)
    raise RuntimeError("POST %s/register failed: %s" % (env_url, last_error))


def poll_work(env_url, server, stop_event):
    while not stop_event.is_set():
        try:
            status, _headers, data = http_request(
                "GET", env_url + "/work", {"accept": "application/json"}, None, WORK_TIMEOUT_S)
            if status == 200:
                work = json.loads(data.decode("utf-8"))
                if is_record(work) and work.get("done") is True:
                    stop_event.set()
                    server.shutdown()
                    return
        except Exception:
            pass
        stop_event.wait(POLL_INTERVAL_S)


def _exit_now(_signum, _frame):
    os._exit(0)


def main(argv):
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    list_mode = "--list" in argv

    providers, ratings, manifest, error = read_brief(prompt_file)
    if error is not None:
        log(error)
    probes = []
    if providers and env_url:
        for provider in providers:
            probe = probe_provider(env_url, provider)
            if not probe[0] and probe[3] is not None:
                log(probe[3])
            probes.append(probe)
    else:
        probes = [(False, 0, [], None)] * len(providers)
    catalog = build_catalog(providers, probes, ratings)

    if list_mode:
        # The same document the register body carries, byte for byte: print
        # the one built object and stop before any server starts.
        sys.stdout.write(json.dumps(catalog, separators=(",", ":")) + "\n")
        sys.stdout.flush()
        return 0

    if not env_url:
        log("MAKEFUN_ENV_URL is not set")
        return 1

    router = Router(env_url, providers, probes, ratings, manifest)
    models = served_model_list(catalog, probes, providers)
    state = AggregatorState(router, models)

    server = ThreadingHTTPServer(("0.0.0.0", 0), make_handler(state))
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        log("the OS picked 8080, which the env owns on one layout: restart the trial")
        return 1
    log("listening on 0.0.0.0:%d with %d providers and %d free models" % (
        port, len(catalog["providers"]), len(models)))
    register(env_url, port, catalog)

    signal.signal(signal.SIGTERM, _exit_now)
    signal.signal(signal.SIGINT, _exit_now)
    stop_event = threading.Event()
    poller = threading.Thread(target=poll_work, args=(env_url, server, stop_event), daemon=True)
    poller.start()
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
