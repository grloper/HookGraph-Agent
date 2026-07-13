# HookGraph

**A supervisor-orchestrated, self-correcting LangGraph ecosystem that turns one long-form
video transcript into three publish-ready vertical clip packages — for TikTok, YouTube
Shorts, and Instagram Reels — with parallel worker fan-out, an adaptive repair memory,
durable human-in-the-loop checkpoints, and an optional Claude-powered creative engine.**

Runs **entirely offline with zero API keys** by default. Every agent boundary is typed,
every retry is checkpointed, and every failure degrades loudly instead of looping forever.

---

## Why this exists

Repurposing a 10–90 minute episode into shorts is the highest-leverage growth activity for
most video creators — and it is still almost entirely manual: someone scrubs the timeline
hunting for "the good parts," guesses at clip boundaries, retypes captions, and rewrites
the title/description/hashtag set three times, once per platform. The platform rules
(hard sub-60s ceilings, cold-open hooks, non-overlapping cuts) are enforced by nothing
but human memory.

HookGraph turns that workflow into a deterministic, auditable **agent newsroom**:

| Agent | Role |
| --- | --- |
| 🎬 **Showrunner** | Hierarchical supervisor. Owns every routing decision via LangGraph `Command`s: kicks off extraction, plans each corrective pass from per-failure **strategy escalation ladders**, and decides when to degrade gracefully or pause for a human. |
| 🔍 **HookExtractor** | Scores every transcript segment for semantic density, emotional spikes, and topic transitions (z-scored against the episode's own baseline), then extracts the **top 3 highest-retention hooks** — each with a virality score, justified rationale, and timestamps snapped to real segment boundaries. On retries it executes exactly the Showrunner's repair plan. |
| ✍️ **Scriptwriter workers** | Not one node — a **parallel worker pool**. One worker per stale hook is fanned out via the `Send` API; each cuts a timestamp-synced caption track (ready-to-burn SRT) and drafts tri-platform metadata. Upsert reducers merge the concurrent writes safely. |
| 🧪 **QualityControl** | A strict, deterministic rubric gate (7 machine-checkable rules). Failures come back as structured violation payloads with remediation hints — never vibes. |
| 📦 **PackageCompiler** | Assembles the final `ClipPackage` deliverables, including an executable ffmpeg render manifest (9:16 crop + subtitle burn-in) per clip. |

The deterministic rubric is the safety rail that makes the generative parts safe: you can
swap any creative engine (titles today; descriptions, CTAs, and hook selection tomorrow)
for an LLM and the loop still *provably converges or degrades* — it can never ship an
invalid clip and never spin forever.

## Architecture

```
                     START
                       │
                       ▼
                ┌─────────────┐  Command(goto=…)
      ┌────────▶│ SHOWRUNNER  │───────────────────────────┐
      │         │ (supervisor)│                           │
      │         └──────┬──────┘                           │
      │    kickoff /   │                                  │  release /
      │    repair plan ▼                                  │  degrade
      │       ┌────────────────┐                          │  (+ optional
      │       │ HOOK EXTRACTOR │                          │   human-review
      │       └────────┬───────┘                          │   interrupt ⏸)
      │                │ Send() fan-out                   ▼
      │                ▼ (1 worker per stale hook)  ┌──────────────────┐
      │   ┌────────────────────────────┐            │ PACKAGE COMPILER │──▶ END
      │   │ ✍️ worker  ✍️ worker  ✍️ worker │            └──────────────────┘
      │   └────────────┬───────────────┘
      │                ▼  (barrier: reducers merge parallel writes)
      │       ┌─────────────────┐
      └───────│ QUALITY CONTROL │
              └─────────────────┘
```

Three patterns make this more than a pipeline:

### 1. Hierarchical supervisor routing (`Command`)

The Showrunner is a real node, not a conditional edge. It returns
`Command(goto=…, update=…)` objects, combining control flow with state writes in one
atomic super-step. All policy — when to retry, which strategy to use, when to give up,
when to summon a human — lives in one auditable place
([`hookgraph/nodes/showrunner.py`](hookgraph/nodes/showrunner.py)).

### 2. Self-correcting repair memory (strategy escalation ladders)

Naive corrective loops repeat the same failed fix until the retry budget dies. HookGraph
keeps a `repair_memory` stream in graph state: every `(hook, rule)` failure maps to an
ordered ladder of increasingly radical strategies, and a strategy that already failed is
**never prescribed twice**:

| Rubric failure | Escalation ladder |
| --- | --- |
| `duration_under_60s` | trim weak edges → tight re-window around the peak → reseed on a new peak |
| `punchy_opening_line` | re-anchor the start → widen the anchor search → reseed on a new peak |
| `non_overlapping_times` | shrink from the collision → shift after the collision → reseed |
| `valid_timestamps` | re-snap boundaries → reseed |
| `exactly_three_hooks` | full tight re-extraction |

When a ladder is exhausted the Showrunner degrades **immediately** — it doesn't burn the
remaining budget on a structurally impossible repair. Every attempted strategy is exported
in the run manifest, so you can audit exactly how the system reasoned its way to a fix.

### 3. Parallel map-reduce workers (`Send` + custom reducers)

Scriptwriter work is embarrassingly parallel, so it runs that way: the dispatch edge fans
out one worker per hook whose artifacts are stale (`(hook_id, revision)`-based staleness,
so an untouched hook is never rewritten). The state channels carry upsert-by-`hook_id`
reducers, which is what makes three concurrent writers merge cleanly — the loop converges
by *repairing state, not rebuilding it*.

### Durable execution & human-in-the-loop

A checkpointer is always attached. Every super-step — every QC retry, every parallel
fan-out, every pause — is snapshotted under the run's `thread_id` and is resumable and
time-travelable via `graph.get_state_history()`. With `--review`, a run that cannot fully
converge stops on a **durable `interrupt()`** and resumes (same thread id) with the
operator's note attached to the shipped packages. Swap in
`langgraph-checkpoint-sqlite`/`-postgres` and the pause even survives a process restart.

### The quality-control rubric

| Rule | Severity | Requirement |
| --- | --- | --- |
| `exactly_three_hooks` | blocker | The package contains exactly 3 clips |
| `duration_under_60s` | blocker | Every clip strictly under 60s (and ≥ 8s) |
| `punchy_opening_line` | blocker | First line clears the punchiness gate (short/interrogative/power opener) |
| `valid_timestamps` | blocker | 0 ≤ start < end ≤ source duration, snapped to segment boundaries |
| `non_overlapping_times` | blocker | No two clips share source footage |
| `metadata_completeness` | blocker | Captions + all 3 platform variants in sync with the hook revision |
| `justified_virality` | warning | Score in (0, 100] with a substantive written justification |

## Quickstart

```bash
git clone https://github.com/grloper/HookGraph-Agent.git
cd HookGraph-Agent

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

That's it — no API keys, no external services (Python 3.10+). The bundled ~12-minute demo
episode runs through the full graph and you watch the corrective loop fire for real:
attempt 1 extracts full narrative arcs that exceed 60 seconds, QualityControl rejects them,
the Showrunner plans a repair, and attempt 2 ships three clips that clear the whole rubric.

Artifacts land in `./output/<video_id>/`: one ready-to-burn `.srt` per clip plus a
`packages.json` manifest containing every hook, caption cue, platform variant, QC report,
repair-memory trace, and ffmpeg render command.

### Run your own transcript

```bash
python main.py --transcript my_episode.json --output-dir dist --max-attempts 4
```

```json
{
  "source_video": { "video_id": "ep-001", "title": "My Episode", "duration_seconds": 900.0 },
  "segments": [
    { "segment_id": 0, "start": 0.0, "end": 8.2, "speaker": "Host", "text": "..." }
  ]
}
```

Exit code `0` = every rubric rule passed; `1` = the run completed degraded and the
packages are flagged for human review.

### Claude-powered creative mode (optional)

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
python main.py --llm
```

`--llm` swaps the titling engine for Claude (structured outputs, `claude-opus-4-8` by
default — override with `HOOKGRAPH_MODEL`). Every call is guarded: a missing key, a
refusal, or any API error falls back to the deterministic engine mid-run, so an LLM
outage can never take the pipeline down. The QC rubric stays deterministic either way —
that is precisely what makes an LLM-in-the-loop safe here.

### Human-review gate (optional)

```bash
python main.py --review
```

If QC cannot fully converge, the run pauses on a durable checkpoint, prints the open
violations, and asks for a reviewer note before the degraded batch ships.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

26 tests cover the scoring engines, the reducer merge semantics, end-to-end convergence,
non-overlap/duration/sync invariants on the final packages, graceful degradation on
impossible inputs, strategy-ladder escalation, the interrupt/resume review gate, and
checkpoint history.

## 🎥 The demo GIF — AI video-generator prompt

> Feed this prompt to your AI video generator of choice to produce the hero demo.

```text
A cinematic 25-second screen-capture-style demo of an AI agent system, 4K, dark-mode
terminal aesthetic with a subtle CRT glow.

SCENE 1 (0–4s): A sleek dark terminal fills the frame. A single command is typed with
soft mechanical key clicks: `python main.py`. On ENTER, a banner slams in:
"HookGraph — supervisor-orchestrated short-form repurposing ecosystem". Camera slowly
dollies in.

SCENE 2 (4–9s): Split screen. LEFT: a long horizontal podcast waveform (12 minutes,
two speakers) scrolling slowly, cool blue. RIGHT: the live terminal streaming agent
events line by line. A glowing amber node labeled SHOWRUNNER pulses at the top of a
graph overlay and fires a directive arrow down to HOOK EXTRACTOR. On the waveform,
three regions ignite in neon green as retention heat-map bars rise over them —
captions flash "emotional spike 88s", "framework 322s", "plot twist 515s".

SCENE 3 (9–14s): The graph overlay animates a fan-out: one node splits into THREE
parallel SCRIPTWRITER worker nodes, each spawning a vertical 9:16 phone mockup.
Word-timed captions type themselves onto each phone in sync with a scrubbing
playhead. Hashtag chips (#Shorts, #fyp, #Reels) snap onto each phone like magnets.

SCENE 4 (14–19s): A red stamp slams onto the middle phone: "QC FAILED — 84s > 60s
ceiling". The waveform region visibly TRIMS itself, weak edge segments shattering
into particles, until a green "58.0s" badge locks in. The terminal prints:
"[Showrunner] Repair plan: trim_weak_edges" then "[QualityControl] Attempt 2
PASSED". A strategy-ladder HUD on the right shows rung 1 of 3 lighting up.

SCENE 5 (19–25s): All three phones align in a row, each stamped "QC APPROVED ✓" in
green. Files materialize below them: three .srt files and packages.json, plus a
scrolling ffmpeg command. Final terminal line types out: "RESULT: pipeline completed
with all rubric rules satisfied." — cut to the HookGraph logotype on black with the
tagline: "One episode in. Three bangers out. Zero babysitting."

Style: Blade-Runner-meets-VS-Code. Deep blacks, neon accents (amber for the
supervisor, cyan for workers, green for QC passes, red for QC failures), smooth
60fps micro-animations, satisfying mechanical typing SFX, a low synth pulse that
resolves on the final stamp.
```

## Repository structure

```
HookGraph-Agent/
├── main.py                        # CLI entry: engine selection, streaming, interrupt handling
├── requirements.txt               # 3 pinned runtime deps — fully offline
├── requirements-dev.txt           # + pytest
├── tests/                         # 26 tests: engines, reducers, e2e, degradation, interrupts
└── hookgraph/
    ├── state.py                   # Pydantic payload models, reducers, graph state
    ├── analysis.py                # deterministic linguistic scoring engines
    ├── engines.py                 # CreativeEngine protocol: deterministic + Claude (fallback-safe)
    ├── routing.py                 # Send() fan-out dispatch for the Scriptwriter worker pool
    ├── graph.py                   # graph compilation + checkpointer wiring
    ├── sample_data.py             # bundled demo episode (offline simulation)
    └── nodes/
        ├── showrunner.py          # supervisor: Command routing, strategy ladders, review gate
        ├── hook_extractor.py      # top-3 hook mining + strategy execution
        ├── scriptwriter.py        # parallel per-hook worker: SRT + tri-platform metadata
        ├── quality_control.py     # the strict rubric gate
        └── package_compiler.py    # final ClipPackage + ffmpeg render manifests
```

## Extending it

- **More creative surface for the LLM** — implement `CreativeEngine` methods for
  descriptions, CTAs, or thumbnail copy in [`hookgraph/engines.py`](hookgraph/engines.py);
  the deterministic rubric keeps the loop safe no matter what the model writes.
- **Real rendering** — each `ClipPackage.render` contains a runnable ffmpeg command
  (9:16 center crop + subtitle burn-in); point it at the source `.mp4` to cut actual clips.
- **More platforms** — add a `PlatformVariant` builder in
  [`hookgraph/nodes/scriptwriter.py`](hookgraph/nodes/scriptwriter.py) and extend the
  `Platform` literal in [`hookgraph/state.py`](hookgraph/state.py); QC's completeness
  check follows the type.
- **New repair strategies** — add a rung to a ladder in
  [`hookgraph/nodes/showrunner.py`](hookgraph/nodes/showrunner.py) and its executor in
  [`hookgraph/nodes/hook_extractor.py`](hookgraph/nodes/hook_extractor.py); the memory
  stream and escalation logic pick it up automatically.
- **Cross-process durability** — pass a `langgraph-checkpoint-sqlite`/`-postgres` saver
  to `build_graph(checkpointer=...)` and review-gate pauses survive restarts.

## License

[MIT](LICENSE)
