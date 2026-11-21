/**
 * Everything the editor knows about mozo's HTTP API.
 *
 * One module, so that the shape of a request lives in one place rather than in whichever component
 * needed it. The node catalogue is passed through as the server sends it -- ordered lists of ports
 * and parameters, derived from the functions that implement them -- and only keyed by name, which
 * is a view rather than a second shape to keep in step.
 */

/**
 * The API under *base*: the address the workflow router is mounted at, ending in `/`.
 *
 * `mount` derives it from where the editor's own script was loaded -- `.../workflow/assets/` is
 * the router's `.../workflow/` -- so the editor finds its API under any prefix with nothing to
 * configure. Every path here is relative to it.
 */
export function client(base) {
    const url = (path) => new URL(path, base).href;

    /**
     * Read a server-sent event stream, calling `onEvent` for each one.
     *
     * Shared by both verbs because the framing is the same either way; what differs is which
     * endpoint and what the events say. `signal` is how a run is cancelled: aborting closes the
     * connection, which closes the generator on the server, which ends the run and closes what it
     * opened.
     */
    async function events(path, form, onEvent, signal) {
        const response = await fetch(url(path), { method: 'POST', body: form, signal });

        if (!response.ok) {
            let detail = `the server refused the workflow (${response.status})`;
            try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
            throw new Error(detail);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        for (;;) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop();
            for (const line of lines) {
                if (line.startsWith('data: ')) onEvent(JSON.parse(line.slice(6)));
            }
        }
    }

    return {
        /**
         * Every node the server offers, keyed by name, and whether it offers a whole-source run.
         *
         * The entries are exactly what `nodes` returns: `{name, category, description, inputs,
         * outputs, parameters}`, where the last three are ordered lists. Nothing is renamed or
         * reshaped -- a translation here would be the second place for it to disagree from.
         */
        async catalogue() {
            const response = await fetch(url('nodes'));
            if (!response.ok) throw new Error(`could not load the node catalogue (${response.status})`);
            const { nodes, process } = await response.json();
            return { nodes: Object.fromEntries(nodes.map(node => [node.name, node])), process };
        },

        /**
         * Test a workflow: one item through the graph, with every node's output drawn on the
         * canvas. `include: 'all'` because the editor draws every node's result.
         */
        stream(document, file, onEvent) {
            return events('stream', body(document, file, { include: 'all' }), onEvent);
        },

        /**
         * Run a workflow over its whole source, calling `onEvent` with `{item}` as each one
         * finishes. No node outputs come back -- a counter and, at most five times a second, one
         * small JPEG of `preview`. Pass an `AbortSignal` to cancel; there is nothing else to call.
         */
        process(document, file, { preview = '', signal } = {}, onEvent) {
            return events('process', body(document, file, { preview }), onEvent, signal);
        },
    };
}

/** The form body every run endpoint takes: the document, an optional file, optional fields. */
function body(document, file, fields) {
    const form = new FormData();
    form.append('workflow', JSON.stringify(document));
    for (const [name, value] of Object.entries(fields)) form.append(name, value);
    if (file) form.append('file', file);
    return form;
}

/**
 * A workflow's name and description, as the document carries them: each only when it says
 * something, so a workflow left unnamed is saved exactly as one from before names existed.
 */
function described(about) {
    const entries = ['name', 'description']
        .map(key => [key, (about?.[key] || '').trim()])
        .filter(([, value]) => value);
    return Object.fromEntries(entries);
}

/**
 * What to call the exported file: the workflow's name, made safe for a file system, or
 * `workflow.json` for one left unnamed.
 */
export function fileNameFor(about) {
    const stem = (about?.name || '').trim().toLowerCase()
        .replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
    return `${stem || 'workflow'}.json`;
}

/**
 * The editor's canvas as a workflow document.
 *
 * Svelte Flow carries every node as its one `default` type with the real kind in `data.nodeType`,
 * because one component draws them all. mozo's format names the kind at the top level. This is the only
 * place that conversion happens, in each direction. *about* is `{name, description}`, the two
 * things a workflow says about itself; {@link fromDocument} hands them back the same way.
 */
export function toDocument(nodes, edges, about = {}) {
    return {
        ...described(about),
        nodes: nodes.map(node => ({
            id: node.id,
            type: node.data.nodeType,
            position: node.position,
            data: { parameters: node.data.parameters || {} },
        })),
        edges: edges.map(edge => ({
            source: edge.source,
            sourceHandle: edge.sourceHandle,
            target: edge.target,
            targetHandle: edge.targetHandle,
        })),
    };
}

/** Does *position* say where to draw a node? `{}` and a missing one do not. */
const placed = position => Number.isFinite(position?.x) && Number.isFinite(position?.y);

/** The space {@link layout} leaves between columns and between rows. */
const COLUMN = 300;
const ROW = 150;

/**
 * Where to draw every node of *document* that has no position, by node id.
 *
 * A document written by code or by hand has none, and Svelte Flow given a node without `x` and
 * `y` draws it nowhere it can be dragged from and routes none of its edges. Positions are layout,
 * not meaning, so they are made up here rather than required: top to bottom by the edges, one row
 * per step from the start (the longest path to it), the nodes of a row side by side. Placed nodes
 * keep their position; when there are some, the rest are laid out to their right.
 */
function layout(document) {
    const nodes = document.nodes || [];
    const loose = nodes.filter(node => !placed(node.position));
    if (!loose.length) return new Map();
    const xs = nodes.filter(node => !loose.includes(node)).map(node => node.position.x);
    const left = xs.length ? Math.max(...xs) + COLUMN : 0;

    const ids = new Set(nodes.map(node => node.id));
    const edges = (document.edges || []).filter(edge => ids.has(edge.source) && ids.has(edge.target));
    const row = new Map(nodes.map(node => [node.id, 0]));
    // At most one pass per node: a longest path cannot be longer, and a cycle (refused by mozo
    // anyway) stops here rather than looping.
    for (let pass = 0; pass < nodes.length; pass++) {
        let moved = false;
        for (const edge of edges) {
            if (row.get(edge.target) <= row.get(edge.source)) {
                row.set(edge.target, row.get(edge.source) + 1);
                moved = true;
            }
        }
        if (!moved) break;
    }

    const taken = new Map();
    return new Map(loose.map(node => {
        const step = row.get(node.id);
        const column = taken.get(step) || 0;
        taken.set(step, column + 1);
        return [node.id, { x: left + column * COLUMN, y: step * ROW }];
    }));
}

/** A workflow document as canvas nodes and edges. The inverse of {@link toDocument}. */
export function fromDocument(document) {
    const laid = layout(document);
    const nodes = (document.nodes || []).map(node => ({
        id: node.id,
        type: 'default',
        position: laid.get(node.id) ?? node.position,
        data: { nodeType: node.type, parameters: node.data?.parameters || {} },
    }));
    const edges = (document.edges || []).map((edge, index) => ({
        id: `${edge.source}-${edge.sourceHandle}-${edge.target}-${edge.targetHandle}-${index}`,
        source: edge.source,
        sourceHandle: edge.sourceHandle,
        target: edge.target,
        targetHandle: edge.targetHandle,
    }));
    const about = { name: document.name || '', description: document.description || '' };
    return { nodes, edges, about };
}

/**
 * Split *edges* into those that can be drawn and those that cannot: an edge can be drawn when the
 * node at each end is among *nodes* and has the port it names, according to *catalogue*.
 *
 * Measured, not guessed: Svelte Flow given an edge to a handle that does not exist stops every
 * store on the page from updating, silently -- the editor that was given it and any other beside
 * it. An edge to a renamed port in an old file, or a typo in a host's document, would otherwise
 * leave a page of dead editors and nothing to say why.
 */
export function drawable(edges, nodes, catalogue) {
    const kinds = new Map(nodes.map(node => [node.id, node.data.nodeType]));
    const has = (id, side, name) => !!catalogue[kinds.get(id)]?.[side]?.some(port => port.name === name);
    const drawn = edges.filter(edge => has(edge.source, 'outputs', edge.sourceHandle)
                                       && has(edge.target, 'inputs', edge.targetHandle));
    return { drawn, left: edges.filter(edge => !drawn.includes(edge)) };
}
