# How to write this library's docs

The documentation of this library is published on https://emrah.fr, next to our other libraries:
Mayaku, Mozo, PixelFlow and Babasu. Write it as plain markdown files in the `docs/` folder of this
repository. The website copies that folder as it is, so the folder structure, the file names and
the header of each file decide everything: the URLs, the menu and the page titles. There is
nothing else to configure.

In the examples below, `<library>` is this library's name in lower case (`mayaku`, `mozo`,
`pixelflow` or `babasu`).

**Rules and suggestions.** This guide has two kinds of content:

- **Rules** (sections 1–3 and 6, marked "Rule"): what the website needs to work, such as the header
  of each page, file names, link and image paths. Follow them exactly.
- **Suggestions** (section 2 "Organising pages into groups", sections 4–5, and the example names and
  folders everywhere): how to organise docs well, not a template. Do not copy the example names
  (`concepts/`, `guides/`, `do-one-thing.md`): pages, groups and titles must come from what this
  library actually has and what its readers look for. Reorganising existing pages is welcome when
  it makes the docs easier to follow.

## 1. Folder structure

An example (the names are placeholders, not required folders or pages):

```
docs/
├── index.md                     → /<library>/                    first page
├── installation.md              → /<library>/installation/
├── concepts/                    → menu group "Concepts"
│   ├── first-concept.md         → /<library>/concepts/first-concept/
│   └── second-concept.md        → /<library>/concepts/second-concept/
├── guides/                      → menu group "Guides"
│   ├── do-one-thing.md          → /<library>/guides/do-one-thing/
│   └── do-another-thing.md
├── reference/                   → menu group "Reference"
│   ├── module-a.md
│   └── module-b.md
└── assets/                      → images and videos (not pages)
    ├── quick-start.png
    └── demo.mov
```

Rules:

- Every `.md` file in `docs/` is one page, whatever its name: a `README.md`, a draft or a note
  left there becomes a public page. Its URL is its path without `.md`:
  `docs/guides/do-one-thing.md` → `/<library>/guides/do-one-thing/`.
- `docs/` holds only pages and the `assets/` folder (plus `llms.txt` if the library has one).
  Example scripts, notebooks and notes for contributors (this guide included) live elsewhere in
  the repository.
- `index.md` is required: it is the library's first page, at `/<library>/`. It exists only
  directly in `docs/`, never inside a sub-folder.
- Sub-folders are optional; each one becomes a menu group. One level only, no folders inside
  folders (except `assets/`). A small library can keep every page directly in `docs/`.
- A page and a folder never share a name (`docs/tracking.md` next to `docs/tracking/`): both would
  claim the same URL.
- File and folder names: lower case, words separated by `-`, no spaces, no numbers for ordering
  (`01-intro.md`). They become URLs, so pick them once and do not rename them: a renamed file is a
  dead link everywhere it was shared.
- The changelog stays in the repository's root `CHANGELOG.md`. Do not copy it into `docs/`.

## 2. The menu

The left menu is built from the files; there is no way to order it by hand.

1. `index.md` first.
2. The other pages directly in `docs/`, alphabetically by file name.
3. Then each sub-folder as a group, groups alphabetically, pages inside alphabetically by file name.

- A group's title is its folder name with `-` turned into spaces and the first letter capitalised:
  `core-concepts/` → "Core concepts".
- The menu shows each page's `label` (see the header below), not its file name.

So when choosing file and folder names, check that the alphabetical order reads well. (In the
example, `concepts/`, `guides/`, `reference/` happen to sort in reading order.)

### Organising pages into groups

Groups are what make the menu readable. A flat list of more than about six pages, sorted
alphabetically, mixes beginner pages with reference pages and reads badly.

- **When:** with more than about six pages, put them in groups. A library with only a handful
  of pages can stay flat.
- **Which groups:** named after what the reader is doing or looking for in *this* library, so
  each name tells them where to click: e.g. `core-concepts/`, `export/`, `models/`,
  `deployment/`, `integrations/`, `api/`. Aim for 2–6 groups of at least 2 pages each. A group
  with a single page usually belongs elsewhere.
- **Top level:** pages directly in `docs/` are listed before every group, so keep there only
  what everyone reads first or looks up anywhere: `index.md`, and if needed pages such as
  `installation.md` or `quick-start.md`. That is how "first steps" come first in the menu without
  a group name that has to sort first.
- **Order:** groups are sorted alphabetically, so look at the resulting menu and, if the order
  is confusing, pick an equally clear name that sorts better (e.g. `api/` or `reference/` both
  work for the last group).

## 3. The header of every page

Rule: every page starts with this header: the page title as a `# Heading` on the first line, then
an HTML comment holding the other fields, one `key: value` per line, all three filled (the values
here are an example):

```markdown
# Count objects in a zone

<!--
summary: Count how many objects enter, leave or stay inside an area, frame by frame.
label: Zone counting
keywords: <library>, zone, counting, object detection
-->

First sentence of the page...
```

| Field         | Shown as                                     | Rule                                                          |
|---------------|----------------------------------------------|---------------------------------------------------------------|
| `# ` heading  | the page heading and the browser tab         | full and specific, under ~55 characters                        |
| `summary`     | the text in search results and link previews | one sentence, under 160 characters, says what the reader gets  |
| `label`       | the entry in the left menu                   | 1–3 words                                                     |
| `keywords`    | search engine keywords                       | the library name first, then 3–6 terms, separated by commas    |

`docs/index.md` has one more line in the comment, exactly this: `template: docs`. No other page
has it.

Each value runs to the end of its line, so colons and quotes inside a value need no escaping.

A page has exactly one `#` heading, its first line. Sections use `##` and `###`.

## 4. Use our libraries first (suggestion)

We maintain Mayaku, Mozo, PixelFlow and Babasu, and the docs of each one lean on the others. When
an example needs another tool (a model, drawing results, reading a video...), use one of ours if
it does the task; their docs at `https://emrah.fr/<library>/` show what each one does.

Use something else only when the page is about it (a page about reading another tool's results
uses that tool) or when our libraries do not do the task. Then say so in one sentence, so the
reader knows the choice is deliberate.

Link to the other libraries' docs with their full URL (`https://emrah.fr/mayaku/`), so the link
also works outside the website.

## 5. Kinds of pages (suggestions)

Most documentation pages are one of these four kinds. Use them as a way to think about what a page
is for, not as a required set: a library does not need all four, and existing pages that work
well stay as they are. What helps the reader is keeping each page to one purpose: a guide does not
turn into a parameter list, and a reference page does not tell a story.

### The first page (`index.md`)

What the library does, how to install it, and a first working result in a few minutes. There is
no separate "Introduction" or "Quick start" page; this page is both. The example below is for a
Python package; for an app, Install is how to download and start it, and Quick start is the first
thing to do in it, with a screenshot.

````markdown
# <Library>

<!--
summary: One sentence: what the library does and for whom.
label: <Library>
keywords: <library>, ...
template: docs
-->

Two or three sentences: what it does, what problem it solves.

## Install

```bash
pip install <library>
```

## Quick start

The shortest example that produces a visible result.

```python
...
```

![What the result looks like](/docs/assets/quick-start.png)

## Next steps

- [First concept](/docs/concepts/first-concept.md): one line on why to read it
- [Do one thing](/docs/guides/do-one-thing.md)
- [Reference](/docs/reference/module-a.md)
````

### Concept pages: one idea per page

Explains something the reader must understand to use the library well: what it is, why it is
shaped that way, how it connects to the rest. Short examples, no full parameter lists.

### Guides: one task per page

A task from start to finish. The title is the task ("Count objects in a zone", not "Zones").

1. One or two sentences: what the reader will have at the end.
2. The steps as `##` headings, each with code that works when pasted in order.
3. The complete code at the end, to copy in one go.
4. The result: an image or a short video.

### Reference: every public function and class

One page per module (for an app: per screen or group of settings). Every entry has the same
layout, so readers can scan:

````markdown
## function_name

```python
function_name(arg, option=None)
```

One sentence: what it does.

| Parameter | Type  | Default | Description |
|-----------|-------|---------|-------------|
| `arg`     | `...` |         | ...         |
| `option`  | `...` | `None`  | ...         |

Returns ...

```python
# a short example
```
````

`##` for each function or class, `###` for the methods of a class.

## 6. Markdown that works

Rule: the site supports standard markdown plus tables. Nothing else (no admonition syntax, no tabs,
no includes, no MDX).

- **Headings:** `##` and `###` are listed in the "On this page" box next to each page, so use
  them for the sections a reader would jump to. `####` is allowed but not listed.
- **Code blocks:** always name the language after the opening fence: `python`, `bash`, `json`,
  `yaml`, or `text` for output. Write shell commands without a `$` prompt, so they can be copied.
- **Tables:** standard pipe tables.
- **Notes and warnings:** a quote with a bold first word:

  ```markdown
  > **Note:** this only works with segmentation models.
  ```

- **Links to other pages:** the path of the page's file from the repository root, starting with
  `/docs/` and ending with `.md`: `[Do one thing](/docs/guides/do-one-thing.md)`. The website
  turns it into the page's URL (`/<library>/guides/do-one-thing/`), and the same link works when
  the docs are read on GitHub or in an editor. Never write the website URL (`/<library>/...`)
  for a page of this library. The website rewrites `/docs/` paths only in links, images and
  `src="..."`, never in plain text or code blocks.
- **Links to a section:** add `#` and the heading's anchor: the heading in lower case, spaces
  turned into `-`, every character other than letters, digits, `-` and `_` removed.
  `## Count objects` → `#count-objects`; `## Detections.filter()` → `#detectionsfilter`:
  `[Install](/docs/index.md#install)`. When two headings on a page have the same text, the
  second one's anchor ends in `-1`, the third in `-2`, and so on.
- **Links to our other libraries:** their full URL, `https://emrah.fr/mayaku/` (see section 4).
- **Images:** put the file in `docs/assets/`, at full size; the site makes the web-sized
  versions. In the page, the path is `/docs/assets/` + the file name:

  ```markdown
  ![Boxes drawn around the cars in a street photo](/docs/assets/quick-start.png)
  ```

  PNG for screenshots and diagrams, JPG for photos; only these two are resized, other formats
  (SVG, GIF...) are published as they are. The text in `[...]` describes the image for someone
  who cannot see it; never leave it empty.
- **Videos:** GitHub cannot play a video from the repository, so each video comes with a
  still image of the same name, next to it in `docs/assets/`:

  ```text
  docs/assets/tracking.mp4   the video, any format (a .mov screen recording is fine)
  docs/assets/tracking.png   a frame of it (PNG or JPG)
  ```

  Draw a play button on the still image, so GitHub readers see it is a video. In the page, show
  the still image and link it to this same page on the website:

  ```markdown
  [![Video: cars tracked across frames](/docs/assets/tracking.png)](https://emrah.fr/<library>/guides/do-one-thing/)
  ```

  On GitHub the reader sees the still image, and clicking it opens the page on the website. On
  the website the link and the image are replaced by the video, with play controls; it starts
  on the video's first frame, not on the still image. The names must match exactly (only the
  extension differs), or the website shows the still image instead. The link must be the
  page's full address on emrah.fr; the build reports one that leads nowhere.

  For a short silent clip that plays by itself in a loop on the website, use this instead. On
  GitHub it shows nothing at all, so use it only where the page reads fine without the clip:

  ```html
  <video autoplay loop muted src="/docs/assets/demo.mov"></video>
  ```

  Keep images small: they are stored in the repository.

## 7. Writing style

- Developer to developer: short sentences, "you", present tense.
- Start with the problem the reader has, then the code that solves it.
- Code first: show the example, then explain what matters in it. Every example works as written,
  imports included. Use realistic names and data, not `foo`/`bar`.
- Say why, not only what: when there are two ways, say when to use which.
- For anything that draws on an image, show the result image. The input image is not needed.
- One idea per page. More than ~5 `##` sections usually means two pages.
- No marketing words ("powerful", "seamless", "blazing fast", "battle-tested"); show a number or
  an example instead. Only numbers you have measured, with what they were measured on.
- Only real uses, no invented scenarios or customers.
- English only.

## 8. Checklist (the rules only)

- [ ] `docs/index.md` exists and has `template: docs` in its header comment
- [ ] `docs/` holds only pages and `assets/` (plus `llms.txt`), with no `README.md` or notes
- [ ] every page starts with its `# ` title, followed by an HTML comment with `summary`, `label`
      and `keywords`, and has no other `#` heading
- [ ] file and folder names are lower case with `-`, one level of sub-folders, no page and folder
      with the same name
- [ ] every code block names its language, and every example works as written
- [ ] every link to another page is `/docs/...md` and points to a file that exists
- [ ] every image and video is in `docs/assets/`, is used by a page, and has a description
- [ ] every video has a still image of the same name, linked to its page on emrah.fr
