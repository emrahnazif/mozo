/**
 * Everything the editor knows about mozo's HTTP API.
 *
 * One module, so that the shape of a request lives in one place rather than in whichever component
 * needed it. The node catalogue is passed through as the server sends it -- ordered lists of ports
 * and parameters, derived from the functions that implement them -- and only keyed by name, which
 * is a view rather than a second shape to keep in step.
 */

import dagre from '@dagrejs/dagre';

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

// --- Documents ------------------------------------------------------------------------------
//
// mozo reads two forms of workflow document (mozo/workflow/document.py). The full form has `edges`,
// each naming both ports. The short form has none: one node per line, its parameters beside its id
// and type, and an input written only where the rule would get it wrong. **The rule:** an input
// not written takes the latest earlier node in the list with an output of its type.
//
// The editor writes the short form and reads both, with the same rule as Python.
// tests/workflow/test_editor_conversion.py runs both on the same documents and compares them.

/** The keys of a short node that are not parameters: mozo.workflow.node.RESERVED. */
const RESERVED = ['id', 'type', 'inputs', 'xy', 'data'];

/** Svelte Flow's node type: one component draws every node, the kind sits in `data.nodeType`. */
const NODE = 'default';

/** The size dagre lays a node out at: every node is 15rem wide, and at most this tall with its
 *  two-line description (see `.svelte-flow__node-default` in editor.css). */
const SIZE = { width: 240, height: 72 };

/** Does *position* say where to draw a node? `{}` and a missing one do not. */
const placed = position => Number.isFinite(position?.x) && Number.isFinite(position?.y);

/** The names of the outputs of a node of *kind* that could feed an input of *type*. */
const outputsOf = (catalogue, kind, type) =>
    (catalogue[kind]?.outputs || []).filter(port => port.type === type).map(port => port.name);

/**
 * `[node, output]` the rule gives an input of *type* after *earlier* (`[{id, kind}]`), or null
 * where it gives none: nothing earlier of the type, or the latest such node giving two of it.
 */
function latest(catalogue, earlier, type) {
    for (let index = earlier.length - 1; index >= 0; index--) {
        const names = outputsOf(catalogue, earlier[index].kind, type);
        if (names.length === 1) return [earlier[index].id, names[0]];
        if (names.length) return null;
    }
    return null;
}

/** An edge for the canvas, with an id of its own. */
const edgeOf = (source, sourceHandle, target, targetHandle, index) => ({
    id: `${source}-${sourceHandle}-${target}-${targetHandle}-${index}`,
    source, sourceHandle, target, targetHandle,
});

/**
 * Where dagre would draw every one of *nodes*, top to bottom along *edges*, by node id.
 *
 * Svelte Flow's own recipe for laying out a graph (svelteflow.dev/examples/layout/dagre), here
 * for a document written by code or by hand, which has no positions: positions are layout, not
 * meaning, so they are made up rather than required.
 */
export function arrange(nodes, edges) {
    const graph = new dagre.graphlib.Graph();
    graph.setGraph({ rankdir: 'TB', nodesep: 40, ranksep: 60 });
    graph.setDefaultEdgeLabel(() => ({}));
    // A copy each: dagre writes the node's place into the object it is given.
    for (const node of nodes) graph.setNode(node.id, { ...SIZE });
    for (const edge of edges) {
        if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) graph.setEdge(edge.source, edge.target);
    }
    dagre.layout(graph);
    return new Map(nodes.map(node => {
        const { x, y } = graph.node(node.id);
        return [node.id, { x: Math.round(x - SIZE.width / 2), y: Math.round(y - SIZE.height / 2) }];
    }));
}

/**
 * A workflow document, in either form, as canvas nodes and edges. *catalogue* is the editor's,
 * `{kind: {inputs, outputs, parameters}}`, which the short form's rule reads port types from.
 *
 * Nothing here refuses a document. An input the rule cannot settle, or a connection to a node or
 * port that is not here, is left unconnected for the person to see and fix; `drawable` and mozo's
 * own checks say what is wrong.
 */
export function fromDocument(document, catalogue = {}) {
    const listed = document.nodes || [];
    const full = 'edges' in document;
    const nodes = listed.map(node => {
        const parameters = full || node.data ? node.data?.parameters || {}
            : Object.fromEntries(Object.entries(node).filter(([key]) => !RESERVED.includes(key)));
        const position = Array.isArray(node.xy) ? { x: node.xy[0], y: node.xy[1] } : node.position;
        return { id: node.id, type: NODE, position, data: { nodeType: node.type, parameters } };
    });

    let edges;
    if (full) {
        edges = document.edges.map((edge, index) =>
            edgeOf(edge.source, edge.sourceHandle, edge.target, edge.targetHandle, index));
    } else {
        const kinds = new Map(listed.map(node => [node.id, node.type]));
        edges = [];
        listed.forEach((node, index) => {
            const earlier = listed.slice(0, index).map(other => ({ id: other.id, kind: other.type }));
            for (const port of catalogue[node.type]?.inputs || []) {
                const written = node.inputs?.[port.name];
                let wire = null;
                if (written === undefined) {
                    wire = latest(catalogue, earlier, port.type);
                } else if (typeof written === 'string' && kinds.has(written)) {
                    const names = outputsOf(catalogue, kinds.get(written), port.type);
                    wire = names.length === 1 ? [written, names[0]] : null;
                } else if (typeof written === 'string') {
                    const dot = written.lastIndexOf('.');
                    wire = dot > 0 ? [written.slice(0, dot), written.slice(dot + 1)] : null;
                }
                if (wire) edges.push(edgeOf(wire[0], wire[1], node.id, port.name, edges.length));
            }
        });
    }

    const loose = nodes.filter(node => !placed(node.position));
    if (loose.length) {
        // Placed nodes keep their position, and the rest go to their right, laid out together.
        const arranged = arrange(nodes, edges);
        const xs = nodes.filter(node => placed(node.position)).map(node => node.position.x);
        const shift = xs.length ? Math.max(...xs) + SIZE.width + 80
            - Math.min(...loose.map(node => arranged.get(node.id).x)) : 0;
        for (const node of loose) {
            const { x, y } = arranged.get(node.id);
            node.position = { x: x + shift, y };
        }
    }

    const about = { name: document.name || '', description: document.description || '' };
    return { nodes, edges, about };
}

/**
 * The editor's canvas as a short workflow document. The inverse of {@link fromDocument}.
 *
 * Each node comes after the nodes it takes from, keeping the canvas's order otherwise, so a small
 * change stays a small change to the file. An input is written only where the rule would get it
 * wrong, and an input with nothing connected is written as `null`, so reading the document back
 * gives the canvas it came from, half-built or not. *about* is `{name, description}`.
 */
export function toDocument(nodes, edges, about = {}, catalogue = {}) {
    const feeding = new Map(edges.map(edge => [`${edge.target}\n${edge.targetHandle}`, edge]));
    const needs = new Map(nodes.map(node => [node.id, new Set(
        edges.filter(edge => edge.target === node.id).map(edge => edge.source))]));
    const waiting = [...nodes];
    const order = [];
    const done = new Set();
    while (waiting.length) {
        // A cycle has no next node; it is kept in canvas order, and mozo says where the cycle is.
        const index = Math.max(0, waiting.findIndex(node => [...needs.get(node.id)].every(id => done.has(id))));
        const [next] = waiting.splice(index, 1);
        order.push(next);
        done.add(next.id);
    }

    const kinds = new Map(nodes.map(node => [node.id, node.data.nodeType]));
    const written = order.map((node, index) => {
        const kind = node.data.nodeType;
        const earlier = order.slice(0, index).map(other => ({ id: other.id, kind: other.data.nodeType }));
        const inputs = {};
        for (const port of catalogue[kind]?.inputs || []) {
            const edge = feeding.get(`${node.id}\n${port.name}`);
            const ruled = latest(catalogue, earlier, port.type);
            if (!edge) {
                inputs[port.name] = null;
            } else if (!ruled || ruled[0] !== edge.source || ruled[1] !== edge.sourceHandle) {
                const names = outputsOf(catalogue, kinds.get(edge.source), port.type);
                inputs[port.name] = names.length === 1 && names[0] === edge.sourceHandle
                    ? edge.source : `${edge.source}.${edge.sourceHandle}`;
            }
        }
        if (!catalogue[kind]) {
            // No catalogue to apply the rule with: every connection is written out, never lost.
            for (const edge of edges.filter(edge => edge.target === node.id)) {
                inputs[edge.targetHandle] = `${edge.source}.${edge.sourceHandle}`;
            }
        }
        const known = catalogue[kind]?.parameters?.map(parameter => parameter.name);
        const parameters = Object.entries(node.data.parameters || {})
            .filter(([name]) => !known || known.includes(name));
        return {
            id: node.id,
            type: kind,
            ...(Object.keys(inputs).length ? { inputs } : {}),
            ...Object.fromEntries(parameters),
            ...(placed(node.position) ? { xy: [Math.round(node.position.x), Math.round(node.position.y)] } : {}),
        };
    });
    return { ...described(about), nodes: written };
}

/** JSON as mozo writes it on one line: `{"a": 1, "b": [2, 3]}`, a space after `:` and `,`. */
function inline(value) {
    if (Array.isArray(value)) return `[${value.map(inline).join(', ')}]`;
    if (value && typeof value === 'object') {
        return `{${Object.entries(value).map(([key, item]) => `${JSON.stringify(key)}: ${inline(item)}`).join(', ')}}`;
    }
    return JSON.stringify(value);
}

/**
 * *document* as text with one node per line, the way mozo saves it, so a text edit -- by a person
 * or an agent -- can replace one node without touching the rest.
 */
export function serialise(document) {
    const head = Object.entries(document).filter(([key]) => key !== 'nodes')
        .map(([key, value]) => `  ${JSON.stringify(key)}: ${inline(value)},`);
    const nodes = (document.nodes || []).map(node => `    ${inline(node)}`);
    const body = nodes.length ? ['  "nodes": [', nodes.join(',\n'), '  ]'] : ['  "nodes": []'];
    return ['{', ...head, ...body, '}'].join('\n') + '\n';
}

/**
 * Connect the inputs of *node*, just dropped onto the canvas, by the short form's rule: each takes
 * the latest node already on the canvas with an output of its type. Returns the edges to add.
 *
 * So dropping the next step of a chain connects it, the way writing the next line of a short
 * document does. An input the rule cannot settle is left for the person to connect.
 */
export function connectDropped(node, nodes, edges, catalogue) {
    const earlier = nodes.filter(other => other.id !== node.id)
        .map(other => ({ id: other.id, kind: other.data.nodeType }));
    return (catalogue[node.data.nodeType]?.inputs || []).flatMap((port, index) => {
        const wire = latest(catalogue, earlier, port.type);
        return wire ? [edgeOf(wire[0], wire[1], node.id, port.name, `${edges.length}-${index}`)] : [];
    });
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
