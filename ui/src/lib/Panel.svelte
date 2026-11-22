<script>
  /**
   * The editor's one panel, beside the canvas: every node to drag on when nothing is selected, the
   * selected node's settings when one is. So the canvas is never covered or moved -- the palette
   * used to sit over its left side and the settings slid in over its right.
   *
   * Drawn into `target` when a host gives `mount` a `panel` element -- a right-hand column of its
   * own, say -- and at the right of the editor's box otherwise. Moved rather than drawn twice, so
   * it stays one component with one state either way.
   */
  import Palette from './Palette.svelte';
  import PropertiesPanel from './PropertiesPanel.svelte';
  import { useState } from './state.js';

  /** Where to draw the panel, or null for beside the canvas. */
  export let target = null;
  export let selectedNode;
  export let updateNodeParameters;
  export let executionResults;

  const { drawerOpen } = useState();

  /** Move the panel into *into*, when there is one; take it back out when the editor goes. */
  function portal(node, into) {
    if (into) into.appendChild(node);
    return { destroy: () => node.remove() };
  }
</script>

<aside class="mz-panel" class:mz-hosted={target} use:portal={target}
       aria-label={$drawerOpen && selectedNode ? 'Node settings' : 'Nodes'}>
  {#if $drawerOpen && selectedNode}
    <button class="mz-back mz-link" on:click={() => drawerOpen.set(false)}>← All nodes</button>
    <PropertiesPanel {selectedNode} {updateNodeParameters} {executionResults} />
  {:else}
    <Palette />
  {/if}
</aside>

<style>
  .mz-panel {
    display: flex;
    flex-direction: column;
    flex: none;
    width: 20rem;
    min-height: 0;
    border-left: 1px solid var(--mz-color-border);
    background: var(--mz-color-surface);
  }

  /* In a host's element, the panel is that element's whole content, at whatever size it is. */
  .mz-panel.mz-hosted {
    width: 100%;
    height: 100%;
    border-left: 0;
  }

  .mz-back {
    flex: none;
    align-self: flex-start;
    padding: var(--mz-space-3) var(--mz-space-3) 0;
  }
</style>
