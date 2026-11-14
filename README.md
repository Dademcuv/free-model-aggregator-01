# Free AI Model Aggregator

A make.fun bounty entry: one server that serves the OpenAI Chat Completions
API and the Anthropic Messages API over ten simulated free-tier model
providers.

For each of the 20 requests the trial sends, the server picks the strongest
free model that meets the request's requirements (tools, vision, JSON mode,
minimum context), speaks that provider's wire (OpenAI, Anthropic or Gemini),
and fails over to another provider when one refuses with a 402 or 429.
Providers are always called non-streaming; streamed client answers are
re-chunked locally from the whole answer, so a stream can never die
mid-failover. The server registers its port and catalog with the trial env
and exits 0 when the workload is done.

`python3 main.py --list` prints the catalog document it registers, byte for
byte. Outside a trial it prints `{"providers":[]}`.

Standard library only; no wall clock and no unseeded randomness in the
request path, so a trial replay leaves the same state export.
