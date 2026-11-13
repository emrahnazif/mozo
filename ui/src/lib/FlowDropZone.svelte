<script>
    import { useSvelteFlow } from '@xyflow/svelte';
    import { createEventDispatcher } from 'svelte';

    const { screenToFlowPosition } = useSvelteFlow();
    const dispatch = createEventDispatcher();

    function onDragOver(event) {
        event.preventDefault();
        event.dataTransfer.dropEffect = 'move';
    }

    function onDrop(event) {
        event.preventDefault();

        const nodeType = event.dataTransfer.getData('application/reactflow');
        if (!nodeType) return;

        // Use the proper screenToFlowPosition from the hook
        const position = screenToFlowPosition({
            x: event.clientX,
            y: event.clientY,
        });

        dispatch('nodedrop', { nodeType, position });
    }
</script>

<div class="mz-drop-zone" role="region" aria-label="Canvas drop zone" on:drop={onDrop} on:dragover={onDragOver}>
    <slot />
</div>

<style>
    .mz-drop-zone {
        width: 100%;
        height: 100%;
    }
</style>