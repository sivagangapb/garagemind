# How GarageMind Works — Explained From Scratch

This is the walkthrough version: what the project does, why it's built the way it is, and how the pieces connect — written so you could explain it to an interviewer (or your future self) without needing to re-read the code first.

## The one-sentence pitch

You tell it a car problem ("P0171" or "rough idle and the check engine light is on"), and it looks that problem up in a structured map of real automotive knowledge, reasons over what it found, and gives you back a diagnosis with sources — instead of just asking an LLM "what's wrong with my car?" and hoping it doesn't make something up.

## The core idea: separate "what's true" from "how it's said"

This is the single most important design decision in the whole project, so it's worth understanding first.

LLMs are great at language and bad at facts they weren't specifically given — ask ChatGPT "what causes P0171" and it'll give you something plausible-sounding, but you have no way to check whether it's actually right or just statistically likely-sounding. That's the classic hallucination problem, and it's a dealbreaker for anything diagnostic.

So GarageMind splits the job into two completely separate pieces:

- **Retrieval decides what's true.** A knowledge graph — a curated, structured database of real OBD-II codes — is walked to find the actual causes, components, and fixes that are connected to what you described. It is *physically incapable* of inventing a cause that isn't in the graph, because it's just graph traversal, not generation.
- **The LLM only phrases it.** Once retrieval has already decided "here are the 3 most likely causes, ranked, with their fixes," the LLM's only job is to turn that into a readable paragraph. It's explicitly instructed to only use what's in front of it.

This pattern is called **RAG** (Retrieval-Augmented Generation), and specifically **GraphRAG** because the retrieval step walks a graph instead of doing a vector/embedding similarity search. Think of it like the difference between asking a random smart person "what's wrong with my car" (pure LLM, can be confidently wrong) versus asking a librarian to pull the exact reference pages on your symptom and then having someone summarize those pages for you (GraphRAG — grounded, and you can go check the pages yourself).

## Step 1 — The data: a curated textbook

Everything starts from `data/dtc_knowledge_base.json`: 30 real SAE J2012 diagnostic trouble codes (the standardized "P0xxx" codes every OBD-II scanner uses), covering fuel/air metering, ignition, emissions, and transmission systems. For each code I wrote out:

- its official description ("System Too Lean (Bank 1)")
- the symptoms a driver would actually notice ("rough idle", "hesitation on acceleration")
- the components involved ("MAF sensor", "vacuum lines")
- a *ranked list of causes*, each tagged high/medium/low likelihood (this matters a lot later)
- the fix sequence a mechanic would actually work through
- which other codes commonly show up alongside this one

This file is the single source of truth. Nothing downstream is allowed to know anything the JSON doesn't say.

## Step 2 — Turning the textbook into a map (the knowledge graph)

A flat JSON file is fine for humans to read but bad for "what connects to what" questions — which is exactly the kind of question a diagnosis needs answered ("if the code is P0171, what causes lead there, and what fixes lead from those causes?").

So `src/graph/build_graph.py` loads that JSON and builds a **graph**: a structure of nodes (things) connected by edges (relationships). Concretely:

```
DTC (P0171) --HAS_SYMPTOM--> Symptom ("rough idle")
DTC (P0171) --INVOLVES-----> Component ("MAF sensor")
DTC (P0171) --CAUSED_BY----> Cause ("vacuum leak")  [edge weight = likelihood]
Cause -------FIXED_BY------> Fix ("smoke-test the intake")
DTC (P0171) --RELATED_TO---> DTC (P0174)   [cross-references from the dataset]
```

I used **NetworkX**, a Python graph library — no database, no vector store, just an in-memory graph object. Building it from the 30 codes produces **354 nodes and 738 edges**. The "high/medium/low" likelihood on each cause becomes a numeric *edge weight* (1.0 / 0.6 / 0.3), which is what lets retrieval later rank causes instead of just listing them.

Why a graph instead of the vector-embedding approach most "RAG" tutorials use? Because the relationships here are categorical and exact, not fuzzy semantic similarity — "P0171 is caused by a vacuum leak" is a fact, not something that benefits from being compared by cosine similarity in embedding space. Graph traversal is also 100% explainable: you can point at the exact path of edges that produced an answer, which is exactly what you want in anything diagnostic.

## Step 3 — GraphRAG retrieval: how it actually looks things up

This is `src/graph/retrieval.py`, and it's the part I'd spend the most time on in an interview, because it's where the interesting engineering problems showed up.

**The basic idea**: given what the user said, find matching "seed" nodes in the graph (an exact DTC code, or a symptom phrase), then walk outward one or two hops to collect the causes, fixes, and related codes connected to those seeds. Rank the causes by (likelihood weight × how much evidence points to them) and return the top few.

**Problem 1 — generic symptoms swallow everything.** Early on, if you gave it an exact code *plus* a sentence like "...and the check engine light is on," the "check engine light" phrase would independently match almost every single code in the graph (because nearly all 30 codes list it as a symptom), and it would pull in a dozen unrelated diagnoses alongside the correct one. The fix: an explicit DTC code is treated as authoritative — the graph trusts it completely and doesn't go symptom-hunting for *more* codes once it has one. And separately, I added an IDF-style filter (the same idea search engines use to downweight the word "the") — any symptom reported by more than ~25% of all codes is considered too generic to seed a new candidate on its own. That single fix took the false-positive rate on ambiguous queries from noisy to clean.

**Problem 2 — how do you resolve a symptom-only query without a code at all?** If someone just says "rough idle," dozens of codes list that. The interesting fix here: when someone describes *multiple* symptoms ("shudder at cruising speed **and** poor fuel economy"), the retrieval step tracks which codes each individual symptom phrase points to, and looks for **agreement across phrases** — a code that's corroborated by two independent symptom clues outranks one only weakly implied by a single generic one. In that exact example, "shudder at cruising speed" is unique to P0740 (torque converter clutch) in the dataset, while "poor fuel economy" alone spans 9 different codes — but because both clues agree on P0740, the system narrows all the way down to that single code instead of returning all 9. That's the closest thing in this project to actual multi-step reasoning, and it's implemented as plain graph/set logic, not an LLM call — cheap, fast, and fully deterministic.

**Problem 3 — when should it just admit it doesn't know?** If retrieval finds nothing, or a symptom-only query is still too broad after the corroboration narrowing (matched more codes than a reasonable threshold), the system is designed to *ask a clarifying question* rather than guess. That routing decision — diagnose vs. ask — is what the LangGraph state machine (next section) is built around.

## Step 4 — The LangGraph agent: the actual "brain," as a flowchart

**LangGraph** lets you define an agent as an explicit graph of steps (nodes) and the rules for moving between them (edges), instead of one long prompt hoping the LLM does the right thing in the right order. Think of it like a flowchart a mechanic actually follows, rather than trusting your gut every time.

GarageMind's agent (`src/agent/graph_agent.py`) has 5 nodes:

1. **`parse_input`** — pulls DTC codes out of what you typed (regex handles the reliable structured part — "P0171" always looks like `P0` + 3 digits — with a light LLM pass alongside it to catch unusual phrasing), and splits free text into rough symptom clauses.
2. **`retrieve_context`** — this is the GraphRAG step from Step 3. It walks the graph and returns a ranked, evidence-backed context block.
3. **A conditional edge** — literally an if/else in the graph itself: if retrieval was empty or still ambiguous, go to `ask_clarification`; otherwise go to `diagnose`. This is the "safety valve" — the agent is structurally prevented from confidently answering when it doesn't actually have enough evidence.
4. **`diagnose`** — the *only* node that calls an LLM for open-ended writing, and its prompt explicitly says: use only the evidence block below, don't invent anything.
5. **`generate_report`** — assembles everything (matched codes, severity, ranked causes, fix sequence, a confidence score derived from how clearly the top cause beat the runner-up, and how many graph nodes the answer is "grounded in") into one structured JSON report.

The reason this is "agentic" rather than just a script: the agent's *path through the graph* changes based on what it finds — a clean single-code query takes a 3-hop path (parse → retrieve → diagnose → report), while an ambiguous one takes a different path (parse → retrieve → clarify) and stops there instead of guessing. That branching, state-carrying control flow is what LangGraph is for.

## Step 5 — The LLM layer, and why it can run for free

`src/llm/provider.py` is a small abstraction so the agent doesn't care which LLM it's talking to. It checks, in order: an explicit choice, then an environment variable, then whichever API key is actually set (OpenAI → Anthropic → Groq), and falls back to an **offline deterministic mock model** if none are set.

That mock model isn't a "TODO, plug in a real key" stub — it actually does the two jobs the real LLM would do, just with simple rules instead of a neural net: it extracts codes with the same regex the real path would fall back on, and it fills in a templated (clearly labeled as templated) diagnosis summary. That's *why* the whole test suite and CLI demo run for $0 and need zero signup — offline mode is a first-class path, not an afterthought, which also happens to make CI trivially free to run.

## Step 6 & 7 — Wrapping it for the real world: FastAPI + n8n

The LangGraph agent by itself is just a Python object — something needs to expose it to the outside world and something needs to decide what happens *after* a diagnosis. Those are two different jobs, so I kept them as two different layers instead of building automation logic into the agent itself:

- **FastAPI** (`src/api/main.py`) exposes one real endpoint, `POST /diagnose`, that takes free text and returns the structured report. This is the seam — anything that can send an HTTP request can now use GarageMind.
- **n8n** (`n8n/garagemind_workflow.json`) is the automation layer sitting on top: a webhook receives the report, calls the FastAPI endpoint, and then *branches* — if the diagnosis needs clarification it just logs it; if it's high severity it fires a notification (email/Slack node) before logging; every call gets appended to a sheet as an audit trail. This is deliberately the kind of workflow a real shop would build: the "smart" part (the agent) stays a clean, testable Python service, and the "business process" part (who gets notified, what gets logged) lives in n8n where it's visual and easy for a non-engineer to modify without touching code.

Keeping these separate is itself a design decision worth naming out loud: it means you could swap n8n for Zapier, or swap the notification channel from email to Slack, without touching a single line of the agent.

## Step 8 — Testing: proving it actually works, not just demoing it

Anyone can show a cherry-picked demo. What's harder to fake is a **test suite that runs against the offline mock model** — 15 tests total, split into:

- graph-construction invariants (every DTC has at least one cause, node counts are right)
- retrieval correctness, including a **regression test for the generic-symptom bug** described above — so if that bug ever crept back in, the test suite would catch it immediately instead of someone noticing months later that diagnoses were getting noisy again
- a **15-case hand-labeled evaluation set** (`tests/eval_set.json`) — realistic inputs paired with the expected outcome, scored for overall accuracy exactly like you'd score a classifier or a retrieval system, with a threshold assertion (currently sitting at 100%, 15/15)

That eval-set pattern — write down what "correct" looks like for a batch of realistic inputs, then measure accuracy against it — is worth calling out specifically in an interview, because it's the actual, honest way to evaluate an LLM system's behavior, as opposed to "I tried it a few times and it looked right."

## Walking through one real example, start to finish

Input: `"P0171 and is running rough at idle"`

1. `parse_input` — regex finds `P0171`. The rest of the sentence is split into clauses; `"is running rough at idle"` becomes a secondary symptom clue.
2. `retrieve_context` — because an explicit code was found, it's treated as authoritative. The graph is walked from `DTC::P0171`: its causes are pulled (vacuum leak — high, dirty MAF sensor — high, weak fuel pump — medium, clogged injector — medium, exhaust leak — low), each with a score, plus the fix sequence and related codes (P0174, P0100, P0300, ...).
3. **Routing** — retrieval wasn't empty and wasn't ambiguous (an explicit code was given), so it goes to `diagnose`, not `ask_clarification`.
4. `diagnose` — the LLM (or the offline mock) receives the ranked evidence block and writes a short summary, told explicitly to only use what's there.
5. `generate_report` — assembles the final JSON: `matched_dtcs: ["P0171"]`, top cause = vacuum leak (highest score), confidence = medium (because vacuum leak and dirty MAF sensor were tied at the top — a real ambiguity worth surfacing), the full recommended fix sequence, related codes to check, and "grounded in 11 graph nodes."
6. If this came in through n8n: it's not high severity (P0171 is "moderate"), so it skips the notification branch and goes straight to being logged, then the report is returned to whoever called the webhook.

## If you get asked "why did you build it this way" in an interview

A few honest, specific answers worth having ready:

*"Why a graph instead of a vector database?"* — Because the relationships in this domain are categorical and exact (a code either has a given cause in the standard or it doesn't), not fuzzy semantic similarity. Graph traversal is also fully explainable — I can point at the exact edges that produced an answer, which matters a lot for anything diagnostic.

*"Why does the LLM only get to phrase things, not decide them?"* — To eliminate hallucination structurally rather than through prompting alone. The agent literally cannot cite a cause that isn't in the graph, because retrieval — not generation — is what selects the causes.

*"What was the hardest bug?"* — The generic-symptom over-matching problem: an explicit code plus an innocuous phrase like "check engine light" was dragging in a dozen unrelated diagnoses, because that symptom is on almost every code. Fixed with two changes — trusting explicit codes over symptom fan-out, and an IDF-style frequency filter on symptoms — and then wrote a regression test so it can't silently come back.

*"How do you know it actually works?"* — A 15-case hand-labeled evaluation set scored for accuracy (currently 100%), plus unit tests on the graph and retrieval layer, all running against a deterministic offline model so the whole suite costs $0 and needs no API key to run in CI.

## Honest limitations (good to say out loud, not hide)

The dataset is 30 curated codes, not the full SAE list — it's sized to prove the architecture works end-to-end, not to be production-ready. Symptom matching is lexical (substring + word overlap), not embedding-based — a deliberate tradeoff for retrieval that's free and fully explainable, at the cost of missing paraphrases an embedding model would catch. And this is a first-pass triage aid, not a replacement for an actual mechanic.
