<script>
    /**
     * One editor: the canvas, the palette, the properties drawer and -- unless the host drives it
     * -- the toolbar.
     *
     * Everything it draws stays inside its own box: nothing is fixed to the window, so a host can
     * put it in any element of any size. What a host can ask of it is exported below and wrapped
     * by `mount` in editor.js: the document in and out, a "change" event, and Test.
     */
    import { createEventDispatcher, onDestroy, onMount } from 'svelte';
    import { get, writable } from 'svelte/store';
    import { Background, SvelteFlow, SvelteFlowProvider } from '@xyflow/svelte';

    import Toolbar from './lib/Toolbar.svelte';
    import CustomNode from './lib/CustomNode.svelte';
    import FlowDropZone from './lib/FlowDropZone.svelte';
    import Palette from './lib/Palette.svelte';
    import PropertiesPanel from './lib/PropertiesPanel.svelte';
    import RunProgress from './lib/RunProgress.svelte';
    import { client, drawable, fileNameFor, fromDocument, toDocument } from './lib/api.js';
    import { createState, noAbout, provideState } from './lib/state.js';

    /** Where the workflow router is mounted, ending in `/`. */
    export let base;
    /** Whether to draw the toolbar. A host that hides it drives the editor through the exports. */
    export let toolbar = true;
    /** The workflow to open, as mozo's document format; none is a fresh canvas. */
    export let document = null;

    const api = client(base);
    const dispatch = createEventDispatcher();
    const state = createState();
    provideState(state);
    const { catalogue, drawerOpen, chosenFile, running, about } = state;

    // Every workflow starts here: pixels have to come from somewhere, and this is the node whose
    // path the runner overrides per image.
    const startingNode = () => ({
        id: 'read_media-1', type: 'default', position: { x: 100, y: 100 },
        data: { nodeType: 'read_media', parameters: {} },
    });

    const nodes = writable([startingNode()]);
    const edges = writable([]);
    const selectedNode = writable(null);
    const executionResults = writable(null);
    const isExecuting = writable(false);
    //: Whether the server offers a whole-source run. Until the catalogue says, it does not.
    let processOffered = false;

    //: The abort handle of the run in progress, or null. Cancelling is aborting: the connection
    //: closes, the server's generator closes with it, and the run ends having closed its files.
    let controller = null;

    //: What the last action was refused for, shown over the canvas rather than in an alert. A
    //: modal stops the editor to say a drag did not take, which the missing edge already said.
    let refusal = null;
    let refusalTimer;

    const nodeTypes = { default: CustomNode };

    //: The document as last reported, so "change" fires for a change to the workflow and not for a
    //: node being selected or measured, which the canvas also writes into its nodes.
    let reported = JSON.stringify(toDocument([startingNode()], [], noAbout()));

    onMount(async () => {
        try {
            const offered = await api.catalogue();
            catalogue.set(offered.nodes);
            processOffered = offered.process;
        } catch (error) {
            say(`Could not load the node catalogue: ${error.message}`);
        } finally {
            ready = true;
            const held = waiting;
            waiting = null;
            if (held) open(held);
        }
    });

    /** Say something went wrong, where the person is looking, and take it back after a while. */
    function say(message) {
        refusal = message;
        clearTimeout(refusalTimer);
        refusalTimer = setTimeout(() => (refusal = null), 6000);
    }

    // --- What a host asks of it ---------------------------------------------------------------

    //: Whether the editor knows its nodes. Until then a document is held, not drawn: its edges have
    //: no handles yet -- the handles come from the catalogue -- and see `drawable` in api.js for
    //: what an edge to no handle does.
    let ready = false;
    let waiting = document;

    /** The workflow on the canvas, as mozo's document format -- or the one waiting to be drawn. */
    export function getDocument() {
        if (waiting) {
            const held = fromDocument(waiting);
            return toDocument(held.nodes, held.edges, held.about);
        }
        return toDocument(get(nodes), get(edges), get(about));
    }

    /** Open *opened* -- a document in mozo's format -- in place of whatever is on the canvas. */
    export function setDocument(opened) {
        if (ready) open(opened);
        else waiting = opened;
    }

    /** Draw *opened*, leaving out every edge whose node or port is not here -- see `drawable`. */
    function open(opened) {
        const canvas = fromDocument(opened);
        const { drawn, left } = drawable(canvas.edges, canvas.nodes, $catalogue);
        if (left.length) {
            say(`Left out ${left.length === 1 ? 'a connection' : `${left.length} connections`} to a `
                + 'node or port that is not here: '
                + left.map(edge => `${edge.source}.${edge.sourceHandle} → ${edge.target}.${edge.targetHandle}`).join(', ')
                + '.');
        }
        nodes.set(canvas.nodes);
        edges.set(drawn);
        about.set(canvas.about);
        selectedNode.set(null);
        executionResults.set(null);
        drawerOpen.set(false);
        reported = JSON.stringify(getDocument());
    }

    $: announce($nodes, $edges, $about);

    function announce() {
        // Once a drag ends rather than on every frame of it: a host saving on "change" would
        // otherwise save sixty times a second.
        if ($nodes.some(node => node.dragging)) return;
        const now = getDocument();
        const serialised = JSON.stringify(now);
        if (serialised === reported) return;
        reported = serialised;
        dispatch('change', now);
    }

    // --- The canvas ---------------------------------------------------------------------------

    function handleNodeDrop(event) {
        const { nodeType, position } = event.detail;
        nodes.update(n => [...n, {
            id: `${nodeType}-${Date.now()}`,
            type: 'default',
            position,
            // A new node sets nothing. Unset means the node's own default, on both sides of the wire.
            data: { nodeType, parameters: {} },
            origin: [0.5, 0.0],
        }]);
    }

    function onNodeClick(event) {
        selectedNode.set(event.detail.node);
        drawerOpen.set(true);
    }

    function onPaneClick() {
        drawerOpen.set(false);
    }

    function onKeydown(event) {
        if (event.key === 'Escape') drawerOpen.set(false);
    }

    //: The editor's own box. Its keys are listened for here and not on the window, so Escape on
    //: the host's page is the host's.
    let root;

    /**
     * Take focus into the editor when it is pressed, so its keys reach it. The canvas cancels the
     * browser's own focusing to drag, so pressing a node would otherwise leave focus on the page.
     * A form control is left to focus itself.
     */
    function focusIn(event) {
        if (!event.target.closest('input, select, textarea, button, label')) root.focus({ preventScroll: true });
    }

    /**
     * Whether a connection may be made, asked by Svelte Flow while the wire is being dragged.
     *
     * Its own hook rather than a check on the `connect` event, because that event is a
     * notification: the library has already added the edge by the time it fires, and returning
     * early from it leaves the edge on the canvas. Asked here, an incompatible target simply does
     * not take the wire -- and it says so while the drag is still happening, by which handles
     * light up, rather than after the fact.
     */
    function isValidConnection(connection) {
        const source = $nodes.find(n => n.id === connection.source)?.data?.nodeType;
        const target = $nodes.find(n => n.id === connection.target)?.data?.nodeType;
        if (!source || !target) return false;

        // The handles the user actually dragged. Only fall back to the first port when a drag did
        // not name one -- picking the first regardless would make a multi-output node's second
        // port unreachable, and would send a handle nobody chose.
        const out = connection.sourceHandle || $catalogue[source]?.outputs?.[0]?.name;
        const into = connection.targetHandle || $catalogue[target]?.inputs?.[0]?.name;
        // Ports are ordered lists of {name, type}, so they are found by name rather than indexed.
        const carried = $catalogue[source]?.outputs?.find(port => port.name === out);
        const taken = $catalogue[target]?.inputs?.find(port => port.name === into);
        return !!carried && !!taken && carried.type === taken.type;
    }

    function updateNodeParameters(nodeId, parameters) {
        const update = node => node && node.id === nodeId
            ? { ...node, data: { ...node.data, parameters } } : node;
        nodes.update(n => n.map(update));
        // And the selection, which the panel reads its current values from. Left stale, the panel
        // spreads the values it was opened with, so setting a second parameter discarded the
        // first.
        selectedNode.update(update);
    }

    /** Mark one node with what a Test is doing to it, or every node with nothing. */
    function mark(nodeId, status) {
        nodes.update(n => n.map(node =>
            nodeId === null || node.id === nodeId ? { ...node, class: status ? `mz-${status}` : '' } : node));
    }

    // --- The verbs ----------------------------------------------------------------------------

    /**
     * Test the workflow on *file* -- an image or a video -- or on the one chosen on its input
     * node. Resolves to `{success, results, error}`: every node's output, keyed by node id.
     */
    export async function test(file = $chosenFile) {
        isExecuting.set(true);
        executionResults.set({ success: true, results: {} });
        mark(null, null);

        try {
            await api.stream(getDocument(), file, (event) => {
                if (event.status === 'running' || event.status === 'completed') {
                    mark(event.node, event.status);
                }
                if (event.status === 'completed') {
                    executionResults.update(r => ({
                        ...r, results: { ...r.results, [event.node]: event.output },
                    }));
                } else if (event.status === 'failed') {
                    if (event.node) mark(event.node, 'failed');
                    executionResults.update(r => ({ ...r, success: false, error: event.error }));
                    say(event.error);
                }
            });
        } catch (error) {
            executionResults.update(r => ({ ...r, success: false, error: error.message }));
            say(error.message);
        } finally {
            isExecuting.set(false);
        }
        return get(executionResults);
    }

    /**
     * Run over the whole source: every frame of a video, every file in a folder.
     *
     * No node outputs come back and none are drawn -- see `RunProgress`. What is watched is
     * whichever node is selected, falling back to a terminal, because the end of the graph is what
     * a person means by "show me" when they have not said.
     */
    async function runWorkflow() {
        const fed = new Set($edges.map(edge => edge.source));
        const watched = $selectedNode?.id || $nodes.filter(node => !fed.has(node.id)).pop()?.id || '';

        controller = new AbortController();
        running.set({ items: 0, seconds: 0, preview: null, node: watched, done: false });
        const began = performance.now();

        try {
            await api.process(getDocument(), $chosenFile, { preview: watched, signal: controller.signal },
                              (event) => {
                if (event.status === 'failed') {
                    say(event.error);
                    running.set(null);
                } else if (event.done) {
                    running.update(r => r && { ...r, items: event.items, seconds: event.seconds, done: true });
                } else {
                    running.update(r => r && {
                        ...r,
                        items: event.item,
                        seconds: (performance.now() - began) / 1000,
                        preview: event.preview || r.preview,
                    });
                }
            });
        } catch (error) {
            // An abort is the Cancel button working, not a failure to report.
            if (error.name !== 'AbortError') {
                say(error.message);
                running.set(null);
            }
        } finally {
            controller = null;
        }
    }

    /** Hang up. That is the whole of cancelling: the server's generator closes with the socket. */
    function cancelRun() {
        controller?.abort();
        running.set(null);
    }

    function exportWorkflow() {
        const blob = new Blob([JSON.stringify(getDocument(), null, 2)], { type: 'application/json' });
        const link = Object.assign(window.document.createElement('a'), {
            href: URL.createObjectURL(blob), download: fileNameFor($about),
        });
        link.click();
        URL.revokeObjectURL(link.href);
    }

    async function importWorkflow(file) {
        try {
            setDocument(JSON.parse(await file.text()));
            dispatch('change', getDocument());      // opened by the person, so it is a change
        } catch (error) {
            say(`That file is not a workflow: ${error.message}`);
        }
    }

    onDestroy(() => {
        clearTimeout(refusalTimer);
        controller?.abort();
    });
</script>

<!-- svelte-ignore a11y-no-noninteractive-tabindex a11y-no-static-element-interactions -->
<div class="mz-editor" tabindex="-1" bind:this={root} on:keydown={onKeydown} on:pointerdown={focusIn}>
    <SvelteFlowProvider>
        {#if toolbar}
            <Toolbar
                onTest={() => test()}
                onRun={processOffered ? runWorkflow : null}
                onCancel={cancelRun}
                onExport={exportWorkflow}
                onImport={importWorkflow}
                testing={$isExecuting}
                runningNow={!!$running && !$running.done}
            />
        {/if}

        <div class="mz-canvas">
            <FlowDropZone on:nodedrop={handleNodeDrop}>
                <SvelteFlow
                    {nodes}
                    {edges}
                    {nodeTypes}
                    proOptions={{ hideAttribution: true }}
                    on:nodeclick={onNodeClick}
                    on:paneclick={onPaneClick}
                    {isValidConnection}
                    maxZoom={1}
                    fitView
                    fitViewOptions={{ maxZoom: 1, padding: 0.2 }}
                >
                    <Background variant="dots" />
                </SvelteFlow>
            </FlowDropZone>

            <Palette />

            <PropertiesPanel
                selectedNode={$selectedNode}
                {updateNodeParameters}
                executionResults={$executionResults}
            />

            <RunProgress />

            {#if refusal}
                <div class="mz-overlay mz-toast" role="status">
                    <span>{refusal}</span>
                    <button class="mz-toast-close" aria-label="Dismiss" on:click={() => (refusal = null)}>×</button>
                </div>
            {/if}
        </div>
    </SvelteFlowProvider>
</div>

<style>
    .mz-editor:focus {
        outline: none;
    }

    .mz-canvas {
        position: relative;
        flex: 1;
        min-height: 0;
    }

    /* Said at the bottom of the canvas, where a rejected drag ended, and dismissable. */
    .mz-toast {
        left: 50%;
        bottom: var(--mz-space-6);
        transform: translateX(-50%);
        max-width: min(40rem, 90%);
    }

    .mz-toast-close {
        border: 0;
        background: none;
        color: inherit;
        font-size: var(--mz-text-md);
        line-height: 1;
        cursor: pointer;
        opacity: 0.7;
    }
</style>
