# AGENTS.md: guide for AI agents working on mozo

mozo runs computer vision models, and workflows built from them, from one `pip install`. Every
model's inference path is vendored into mozo and verified bit-identical to the original. Its
users are developers who install it from PyPI. Every decision here is judged by what is best
for them.

## Ownership

One agent session maintains each project, and only that session changes its repo.
You are the owner and long-term maintainer of this codebase, and its design is up to you.

- You may read other repos, for example to see how they use mozo or how a PixelFlow API works.
  Never edit them, commit in them, or run git commands that write there.
- Find another project's session with `ListAgents`.
- mozo depends on PixelFlow's released versions from PyPI (`pixelflow>=...` in
  `pyproject.toml`), never on a local checkout's unreleased code.

## Requests between projects

Requests go both ways: other projects (such as babasu) ask mozo for things, and mozo asks
PixelFlow for things. Always with `SendMessage` to the owning session. Never work around a
missing feature by editing the other repo or copying its code into mozo.

A request says:

1. **What** is needed, as the caller would write it (`video.frame_time`, not "timing info").
2. **Why**: the user-facing problem it solves, not only the requester's own workflow.
3. **Proposed API**, if any, and what the requester does today without it.

When mozo receives a request:

- **Judge it only by mozo's users.** Would a typical mozo user want this? Accept, propose a
  more general version, or refuse with a reason. A need only one project has stays in that
  project.
- **Prefer the general fix** over the one shaped around the requester's workflow.
- **New or changed public API needs the user's approval** before it is committed. Removing
  public API is a breaking change.
- **Reply** with the decision, and later with the version that ships it.

When mozo needs something from PixelFlow, ask and wait for the release. Don't ship mozo code
that depends on unreleased PixelFlow API.

## Rules

- Images go in and out through `mozo/image.py` (`load_image`, `encode_image`), which hands
  decoding and encoding to PixelFlow, the one owner of the colour-channel conversion. Never
  add a second encoder or decoder.
- Never commit, push, tag or release without the user asking. No `Co-Authored-By` lines.
- Changes go under `## Unreleased` at the top of `RELEASE_NOTES.md`, in plain language.
- Keep the public API small: adding API later is easy, removing it breaks users.

## Plans

Work that takes more than a session, touches public API or adds a model family starts with a
plan in `plans/<name>.md`, written before the code. `plans/` is gitignored: plans are working
notes, so nothing in the repo may depend on one. For a model family, follow `VENDORING.md`.
Either way:

- **Check the code first.** Start with what already exists and what blocks the work, with
  `file:line` references.
- **Measure, don't assert.** Every number is either measured, with the machine it came from, or
  marked open.
- **Scope first.** Say what is out, and why, before what is in.
- **Number the decisions**, each with its reason, and record the user's approval:
  `## Decided (approved YYYY-MM-DD)`.
- **Order the phases** so each is useful on its own and checked as soon as it exists.
- **Write results back into the plan** when the work lands, especially what reading didn't
  predict.

## Before you finish

```bash
python -m pytest tests/ -q --ignore=tests/families
```

All tests must pass. Show the result; don't just say it passed.

## Releasing (only when the user asks)

1. Set `__version__` in `mozo/__init__.py`.
2. In `RELEASE_NOTES.md`, turn the unreleased notes into a `# Mozo vX.Y.Z: <title>` section:
   the publish workflow takes the release notes from that heading.
3. Run the tests, commit, then tag `vX.Y.Z`. The workflow checks the tag matches the version.
4. Pushing the tag publishes to PyPI and cannot be undone: the user decides when.

## Stop and ask the user

- New, changed or removed public API.
- Releases, tags, pushes, version numbers.
- New dependencies, new model families, or licence changes (weights licences included).
- Deleting or rewriting files or commits you did not make.
