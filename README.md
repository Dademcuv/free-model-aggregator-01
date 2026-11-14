# Free AI Model Aggregator

Entry for make.fun's `free-model-aggregator-01` bounty: one server, written
in Python 3.13 with the standard library only, that serves the OpenAI Chat
Completions API and the Anthropic Messages API (JSON and server-sent events)
over ten simulated free-tier providers speaking the real OpenAI, Anthropic
and Gemini wires.

## How it works

- Reads the trial brief (`$MAKEFUN_PROMPT_FILE`): the provider directory,
  the quality table and the workload manifest.
- Probes every provider's catalog route in its own wire and builds one
  catalog document — the same document `python3 main.py --list` prints and
  the register body posts.
- Routes each request to the highest-rated free model that meets the
  request's requirements (tools, vision, JSON mode, context length),
  translating the request into the provider's wire and the answer back into
  the client's wire, preserving the system prompt, user text, images, tools,
  stop sequences, `max_tokens` and JSON mode exactly.
- Fails over to the next candidate when a provider refuses (402/429/5xx);
  a refusal penalises that provider for later requests without writing it
  off, since budgets are hidden and a refusal may be a scheduled failure.
- Calls providers non-streaming and re-chunks streamed client answers
  locally, so a stream can never die half-way through a failover.
- Polls the env's `/work` route and exits 0 when the workload is done.

Everything is deterministic: every choice is a function of the brief, the
requests and the providers' answers. No wall clock, no randomness.

## Files

- `main.py` — the whole server (single file, standard library only).
- `LICENSE` — MIT.

## Run

```sh
python3 main.py --list   # print the catalog document and exit 0
python3 main.py          # serve a trial (needs $MAKEFUN_ENV_URL and $MAKEFUN_PROMPT_FILE)
```
