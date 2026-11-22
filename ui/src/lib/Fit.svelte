<script>
    /**
     * Fits the canvas to the graph each time `request` changes: after a document is opened and
     * after Tidy up, not only on the first load, which is all Svelte Flow's `fitView` prop does.
     *
     * A component of its own because `useSvelteFlow` only works inside the provider, and the
     * editor's root is the component that draws the provider.
     */
    import { useSvelteFlow } from '@xyflow/svelte';

    export let request = 0;
    const { fitView } = useSvelteFlow();

    // Two frames: one for the nodes to be drawn, one for them to be measured. Fitting before
    // that fits nodes of no size.
    $: if (request) requestAnimationFrame(() => requestAnimationFrame(() =>
        fitView({ maxZoom: 1, padding: 0.2 })));
</script>
