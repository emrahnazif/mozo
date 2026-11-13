<script>
    /**
     * The drawer with one node's parameters, and what it produced in the last Test.
     *
     * Every widget comes from the `kind` the server declared -- `int`, `float`, `str`, `bool`,
     * `color`, or `select` with its options. Nothing here guesses from the value, which is the
     * point of the catalogue being derived from the node's own signature: a parameter that has
     * never been set still knows what it is.
     *
     * A parameter marked `optional` may be left blank, and blank is sent as nothing rather than as
     * a zero or an empty string. That is what lets a thickness scale itself to the image and a
     * crop run to the image's own edge.
     *
     * `source` is the one kind that is not just a box: the value is a path, and a person at a
     * browser cannot write the server's path to a file sitting on their own machine. So it offers
     * a file to upload as well, and says which of the two the next run will use. `path` is a file
     * the workflow keeps as saved -- a checkpoint, annotations to read back -- so a box alone.
     *
     * `list` is several strings, one per line, so an entry may hold a comma: a prompt like
     * "bolt, M8" stays one prompt. A workflow saved before these were lists holds a
     * comma-separated string; it is shown split into lines, so editing it keeps what it meant.
     */
    import { useState } from './state.js';

    export let selectedNode;
    export let updateNodeParameters;
    export let executionResults;

    const { catalogue, chosenFile, drawerOpen } = useState();

    $: nodeInfo = $catalogue[selectedNode?.data?.nodeType] || {};
    $: fields = nodeInfo.parameters || [];
    $: set = selectedNode?.data?.parameters || {};
    $: result = executionResults?.results?.[selectedNode?.id];

    /** snake_case as words: `max_hole_area` is "Max hole area". */
    const label = (name) => name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, ' ');

    /** A value as a box shows it: unset is blank. */
    function shown(value) {
        return value === null || value === undefined ? '' : value;
    }

    /** What a field is showing: what the workflow saved, or the node's own default. */
    function valueOf(field) {
        return field.name in set ? set[field.name] : field.default;
    }

    function change(field, value) {
        updateNodeParameters(selectedNode.id, { ...set, [field.name]: value });
    }

    /** An empty box on an optional field means "unset", which is a value the server understands. */
    function changeNumber(field, raw, whole) {
        if (raw === '' && field.optional) return change(field, null);
        const parsed = whole ? parseInt(raw, 10) : parseFloat(raw);
        change(field, Number.isNaN(parsed) ? field.default : parsed);
    }

    function changeText(field, raw) {
        change(field, raw === '' && field.optional ? null : raw);
    }

    /**
     * One entry per line: a list as it is, an old comma string split the way the server splits it
     * (`comma_separated` in mozo/text.py), so the two cannot read it differently.
     */
    function lines(value) {
        if (Array.isArray(value)) return value.join('\n');
        if (typeof value === 'string') return value.split(',').map((p) => p.trim()).filter(Boolean).join('\n');
        return '';
    }

    /** Each non-empty line is one entry, kept exactly as typed -- a comma included. */
    function changeLines(field, raw) {
        change(field, raw.split('\n').map((p) => p.trim()).filter(Boolean));
    }

    /** Take the first file from a picker or a drop, whichever it arrived by. */
    function chooseFile(files) {
        if (files?.[0]) chosenFile.set(files[0]);
    }

    let dragging = false;
</script>

<aside class="mz-card mz-drawer mz-rows" class:mz-open={$drawerOpen} aria-label="Node properties">
    {#if selectedNode}
        <div class="mz-section">
            <p class="mz-title">{selectedNode.data.nodeType}</p>
            <p class="mz-hint">{nodeInfo.description || ''}</p>
        </div>

        {#each fields as field (field.name)}
            {@const value = valueOf(field)}
            <div class="mz-section">
                {#if field.kind === 'bool'}
                    <label class="mz-check">
                        <input id={field.name} type="checkbox" checked={!!value}
                               on:change={(e) => change(field, e.target.checked)} />
                        {label(field.name)}
                    </label>
                {:else}
                    <label class="mz-label" for={field.name}>
                        {label(field.name)}
                        {#if field.optional}<span class="mz-muted"> — optional</span>{/if}
                    </label>
                    {#if field.kind === 'select'}
                        <select id={field.name} class="mz-select" value={value}
                                on:change={(e) => change(field, e.target.value)}>
                            {#each field.options as option}
                                <option value={option}>{option}</option>
                            {/each}
                        </select>
                    {:else if field.kind === 'color'}
                        <div class="mz-row">
                            <input id={field.name} class="mz-color" type="color" value={value || '#00ff00'}
                                   on:input={(e) => change(field, e.target.value)} />
                            {#if field.optional}
                                <button class="mz-link" on:click={() => change(field, null)}>by class</button>
                            {/if}
                        </div>
                    {:else if field.kind === 'list'}
                        <textarea id={field.name} class="mz-textarea" rows="3"
                                  value={lines(value)} placeholder="one per line"
                                  on:input={(e) => changeLines(field, e.target.value)}></textarea>
                        <p class="mz-hint">One per line.</p>
                    {:else if field.kind === 'path'}
                        <input id={field.name} class="mz-input" type="text" value={shown(value)}
                               placeholder="a path on the server"
                               on:input={(e) => changeText(field, e.target.value)} />
                    {:else if field.kind === 'source'}
                        <div class="mz-drop" class:mz-dragging={dragging} role="button" tabindex="-1"
                             on:dragover|preventDefault={() => (dragging = true)}
                             on:dragleave={() => (dragging = false)}
                             on:drop|preventDefault={(e) => {
                                 dragging = false;
                                 chooseFile(e.dataTransfer.files);
                             }}>
                            <label class="mz-button mz-full">
                                {$chosenFile ? $chosenFile.name : 'Drop or choose a file'}
                                <input type="file" hidden on:change={(e) => chooseFile(e.target.files)} />
                            </label>
                        </div>
                        {#if $chosenFile}
                            <p class="mz-hint">
                                The next run reads this file.
                                <button class="mz-link" on:click={() => chosenFile.set(null)}>
                                    Use the path instead
                                </button>
                            </p>
                        {/if}
                        <input id={field.name} class="mz-input mz-gap" type="text" value={shown(value)}
                               placeholder="or a path on the server -- a file or a folder"
                               disabled={!!$chosenFile}
                               on:input={(e) => changeText(field, e.target.value)} />
                        {#if !$chosenFile}
                            <p class="mz-hint">
                                A folder runs every image in it. Folders are server paths --
                                only single files can be uploaded.
                            </p>
                        {/if}
                    {:else if field.kind === 'int' || field.kind === 'float'}
                        <input id={field.name} class="mz-input" type="number"
                               step={field.kind === 'int' ? '1' : '0.05'}
                               value={shown(value)}
                               placeholder={field.optional ? 'automatic' : ''}
                               on:input={(e) => changeNumber(field, e.target.value, field.kind === 'int')} />
                    {:else}
                        <input id={field.name} class="mz-input" type="text" value={shown(value)}
                               placeholder={field.optional ? 'unset' : ''}
                               on:input={(e) => changeText(field, e.target.value)} />
                    {/if}
                {/if}
            </div>
        {/each}

        {#if result}
            <div class="mz-section">
                <p class="mz-label">Result</p>
                {#if typeof result === 'string' && result.startsWith('data:image')}
                    <img src={result} alt="what this node produced" />
                {:else if result && result.depth}
                    <img src={result.depth} alt="depth map" />
                    <p class="mz-hint">{result.min.toFixed(2)} to {result.max.toFixed(2)}</p>
                {:else if Array.isArray(result)}
                    <p class="mz-hint">{result.length} results</p>
                    <pre>{JSON.stringify(result.slice(0, 10), null, 1)}</pre>
                {:else}
                    <pre>{JSON.stringify(result, null, 1)}</pre>
                {/if}
            </div>
        {/if}
    {:else}
        <p class="mz-section mz-muted">Select a node to edit it.</p>
    {/if}
</aside>

<style>
    /* Slides in from the right edge of the canvas when a node is selected. */
    .mz-drawer {
        top: var(--mz-space-4);
        right: var(--mz-space-4);
        width: 20rem;
        max-height: calc(100% - 2 * var(--mz-space-4));
        overflow-y: auto;
        transform: translateX(calc(100% + 2 * var(--mz-space-4)));
        transition: transform var(--mz-duration-slow) var(--mz-ease-out);
    }

    .mz-drawer.mz-open {
        transform: none;
    }

    .mz-section {
        padding: var(--mz-space-3);
    }

    .mz-check {
        display: flex;
        align-items: center;
        gap: var(--mz-space-2);
        font-weight: var(--mz-weight-medium);
    }

    .mz-row {
        display: flex;
        align-items: center;
        gap: var(--mz-space-2);
    }

    .mz-color {
        width: 3rem;
        height: 2rem;
        padding: 0;
        border: 0;
        background: none;
    }

    .mz-gap {
        margin-top: var(--mz-space-2);
    }

    /* The drop target. Dashed while idle so it reads as "put something here" rather than as a
       button that has lost its border, and filled while a file is over it. */
    .mz-drop {
        padding: var(--mz-space-1);
        border: 1px dashed var(--mz-color-border);
        border-radius: var(--mz-radius-md);
        transition: border-color var(--mz-duration-fast), background-color var(--mz-duration-fast);
    }

    .mz-drop.mz-dragging {
        border-color: var(--mz-color-highlight);
        background: var(--mz-color-accent-bg);
    }

    img {
        display: block;
        max-width: 100%;
        border-radius: var(--mz-radius-md);
    }

    pre {
        max-height: 12rem;
        overflow: auto;
        font-family: var(--mz-font-mono);
        font-size: var(--mz-text-xs);
    }
</style>
