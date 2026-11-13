<script>
    /**
     * What a run in progress looks like: how far it has got, and a glance at one node.
     *
     * Not the canvas. Test draws every node's output at full size on the graph itself; a run has
     * ten thousand of those and no room for any of them, so this is one counter and one small
     * picture in a corner. The picture is a JPEG a fraction the size of the ones the canvas gets,
     * arriving at most five times a second however fast the run goes.
     */
    import { useState } from './state.js';

    const { running } = useState();

    $: rate = $running?.seconds ? Math.round($running.items / $running.seconds) : null;
</script>

{#if $running}
    <aside class="mz-overlay mz-run" aria-live="polite">
        {#if $running.preview}
            <img src={$running.preview} alt="the run in progress" />
        {/if}
        <div class="mz-tally">
            <strong>{$running.items.toLocaleString()}</strong>
            <span>{$running.done ? 'items' : 'items so far'}</span>
            {#if rate}<span class="mz-faint">{rate.toLocaleString()}/s</span>{/if}
            {#if $running.node}<span class="mz-faint mz-node-name">showing {$running.node}</span>{/if}
        </div>
    </aside>
{/if}

<style>
    .mz-run {
        bottom: var(--mz-space-4);
        right: var(--mz-space-4);      /* the palette runs down the left */
        max-width: min(28rem, calc(100% - 2 * var(--mz-space-4)));
        padding: var(--mz-space-2);
    }

    .mz-run img {
        display: block;
        width: 10rem;
        max-width: 40%;
        border-radius: var(--mz-radius-md);
    }

    .mz-tally {
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
        font-size: var(--mz-text-xs);
        /* Digits that change every frame must not shuffle the layout under them. */
        font-variant-numeric: tabular-nums;
    }

    .mz-tally strong { font-size: 1.15rem; }
    .mz-faint { opacity: 0.65; }
    .mz-node-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
