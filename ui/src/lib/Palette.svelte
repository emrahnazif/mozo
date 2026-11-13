<script>
  /**
   * Every node the server offers, to drag onto the canvas, with a search box over them.
   *
   * The list is the catalogue as the server sends it, which is the registry: a node a host
   * registers with `@node` appears here with nothing else to do.
   */
  import CategoryIcon from './CategoryIcon.svelte';
  import Icon from './Icon.svelte';
  import { useState } from './state.js';

  const { catalogue } = useState();
  let query = '';

  $: needle = query.trim().toLowerCase();
  $: shown = Object.entries($catalogue).filter(([name, info]) => !needle
      || [name, info.description, info.category].some(text => (text || '').toLowerCase().includes(needle)));

  function dragStart(event, name) {
    event.dataTransfer.setData('application/reactflow', name);
    event.dataTransfer.effectAllowed = 'move';
  }
</script>

<aside class="mz-card mz-palette" aria-label="Nodes">
  <div class="mz-search">
    <Icon name="magnifying-glass" />
    <input class="mz-input" type="search" placeholder="Search nodes" aria-label="Search nodes"
           bind:value={query} />
  </div>
  <ul class="mz-list mz-rows">
    {#each shown as [name, info] (name)}
      <li class="mz-entry mz-item" draggable="true" on:dragstart={(event) => dragStart(event, name)}>
        <span class="mz-entry-icon"><CategoryIcon category={info.category} /></span>
        <div>
          <p class="mz-title">{name}</p>
          <p class="mz-entry-about">{info.description || ''}</p>
        </div>
      </li>
    {:else}
      <li class="mz-empty mz-muted">{needle ? 'No node matches.' : 'Loading nodes…'}</li>
    {/each}
  </ul>
</aside>

<style>
  .mz-palette {
    top: var(--mz-space-4);
    left: var(--mz-space-4);
    width: 18rem;
    max-height: calc(100% - 2 * var(--mz-space-4));
  }

  .mz-search {
    position: relative;
    flex: none;
    padding: var(--mz-space-3);
    border-bottom: 1px solid var(--mz-color-border);
    color: var(--mz-color-text-muted);
  }

  .mz-search :global(.mz-icon) {
    position: absolute;
    top: 50%;
    left: calc(var(--mz-space-3) + var(--mz-space-2));
    transform: translateY(-50%);
    pointer-events: none;
  }

  .mz-search .mz-input {
    padding-left: calc(var(--mz-space-2) * 2 + 1rem);
  }

  .mz-list {
    margin: 0;
    padding: 0;
    overflow-y: auto;
    list-style: none;
  }

  .mz-item {
    cursor: grab;
  }

  .mz-item:hover {
    background: var(--mz-color-accent-bg);
  }

  .mz-empty {
    padding: var(--mz-space-3);
  }
</style>
