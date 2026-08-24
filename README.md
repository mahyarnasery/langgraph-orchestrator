# LangGraph Orchestrator

A local multi-model coding-agent orchestration system built with
[LangGraph](https://github.com/langchain-ai/langgraph) and
[Ollama](https://ollama.com/).

The project explores how specialized local LLMs can be coordinated to
plan, execute, review, and iteratively improve software changes while
minimizing model switching and unnecessary context.

## Architecture

The orchestrator uses three specialized model roles:

| Role | Model | Responsibility |
|---|---|---|
| Planner | `gemma4:e4b` | Analyze the repository, define objectives, architecture, scope, and acceptance criteria |
| Foreman | `ibm/granite4.1:8b` | Turn the plan into executable task batches and coordinate implementation |
| Worker | `ibm/granite4.1:3b` | Execute individual implementation tasks using repository tools |

The roles intentionally have different responsibilities.

The Planner is not an implementation agent. It reasons about what should
be done.

The Foreman translates that reasoning into concrete implementation work
and coordinates execution.

The Worker performs the actual file-level changes.

## Why multiple models?

The system is designed for local execution on constrained hardware.

The available GPU has limited VRAM, so keeping all models resident
simultaneously is impractical. The orchestrator therefore loads models
sequentially and uses task batching to reduce unnecessary model
switching.

The Worker also has a substantially smaller practical context budget than
the larger reasoning model. Tasks are therefore structured and batched
carefully so that the Worker receives only the context needed for its
current work.

## Core Design Principles

### 1. Authority boundaries

Different layers have different permissions.

- Planner: reasoning and planning
- Foreman: task decomposition, integration, and repository preparation
- Worker: controlled implementation through repository tools

The Worker is intentionally not given unrestricted repository authority.

### 2. Structured state

Communication between orchestration layers uses structured state and
Pydantic schemas rather than passing unrestricted conversational history
between models.

### 3. Context-aware batching

The Foreman groups related implementation tasks into a `TaskBatch`.

The Worker can then execute multiple related tasks during one model
session, reducing model loading/unloading overhead and avoiding repeated
context initialization.

### 4. Evidence-based execution

Later orchestration cycles use evidence from previous execution rather
than assuming that earlier tasks succeeded.

Execution results, tool outcomes, and repository state are used to inform
subsequent planning and coordination.

## Project Milestones

### Alpha01

Alpha01 established the structured orchestration state and introduced
previous-cycle memory and batched Worker execution.

Key goals:

- Structured state between orchestration layers
- Task batching
- Previous-cycle state
- Context-aware execution
- Sequential local model loading

Tag:

`alpha01`

### Alpha02

Alpha02 extended the orchestration loop with evidence-based execution
and task batching.

The Foreman produces executable task batches, while Worker results and
tool outcomes are fed back into the orchestration state so that later
cycles can reason from actual execution evidence.

Tag:

`alpha02`

Current Alpha02 work also established the basis for iterative cycles in
which planning can react to what actually happened during previous
execution.

## Current Status

The project currently has a working three-layer LangGraph orchestration
prototype with:

- Planner → Foreman → Worker execution
- Ollama-backed local models
- Structured Pydantic state
- Task batching
- Previous-cycle state
- Worker tool calling
- Evidence-based execution
- Sequential model loading/offloading
- Git-tracked Alpha milestones

The current implementation is still experimental.

In particular, the latest execution tests exposed an important
orchestration issue: the Worker can be asked to modify files that have
not yet been created by the Foreman. This demonstrates why repository
state and task dependencies must be explicitly handled by the
orchestration layer rather than assumed by the Worker.

## Repository Structure

```text
langgraph-orchestrator/
├── src/
│   └── orchestration/
│       ├── graph.py
│       ├── models.py
│       ├── schemas.py
│       ├── state.py
│       └── tools.py
├── .gitignore
├── langgraph.json
└── README.md
