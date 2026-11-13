<script>
  /**
   * What you do to a workflow as a whole: name it, test it, run it, save it, open one.
   *
   * **Two verbs, because they are two different things.** Test puts one item through the graph and
   * draws every node's output, which is what you want while wiring. Run puts the whole source
   * through and draws none of them -- see `process` in api.js for why it cannot. Run is offered
   * only where the server offers it: a host that runs workflows as jobs of its own leaves it out.
   *
   * At the top of the editor's own box rather than fixed to the window, so it can never sit over a
   * host's header. A host that would rather use its own buttons hides it: `toolbar: false` to
   * `mount`, or `?toolbar=0` on the page.
   *
   * The file a run reads is not here. It belongs to the node that reads it, where the person is
   * already looking when they wonder what it will run on -- see the `source` widget in
   * PropertiesPanel.
   */
  import Icon from './Icon.svelte';
  import { useState } from './state.js';

  export let onTest;
  /** Null where the server does not offer a whole-source run. */
  export let onRun;
  export let onCancel;
  export let onExport;
  export let onImport;
  export let testing;
  export let runningNow;

  const { about } = useState();
  let picker;

  function picked(event) {
    const [file] = event.target.files;
    if (file) onImport(file);
    event.target.value = '';      // so choosing the same file again still counts
  }
</script>

<header class="mz-toolbar">
  <div class="mz-about">
    <input class="mz-name" type="text" placeholder="Untitled workflow"
           aria-label="Workflow name" bind:value={$about.name} />
    <input class="mz-description" type="text" placeholder="What it does"
           aria-label="Workflow description" bind:value={$about.description} />
  </div>

  <div class="mz-actions">
    <button class="mz-button" on:click={onTest} disabled={testing || runningNow}
            title="One item through the graph, with every node's output shown">
      <Icon name="play" />{testing ? 'Testing…' : 'Test'}
    </button>

    {#if onRun}
      {#if runningNow}
        <button class="mz-button mz-destructive" on:click={onCancel}>
          <Icon name="stop" />Cancel
        </button>
      {:else}
        <button class="mz-button mz-primary" on:click={onRun} disabled={testing}
                title="Every item the input produces">
          <Icon name="forward" />Run
        </button>
      {/if}
    {/if}

    <button class="mz-button" on:click={onExport} title="Save the workflow as a file">
      <Icon name="arrow-down-tray" />Export
    </button>
    <button class="mz-button" on:click={() => picker.click()} title="Open a workflow file">
      <Icon name="arrow-up-tray" />Import
    </button>
    <input bind:this={picker} type="file" accept=".json,application/json" hidden on:change={picked} />
  </div>
</header>

<style>
  .mz-toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--mz-space-4);
    flex: none;
    height: var(--mz-toolbar-height);
    padding: 0 var(--mz-space-4);
    border-bottom: 1px solid var(--mz-color-border);
    background: var(--mz-color-bg);
  }

  .mz-about {
    display: flex;
    flex-direction: column;
    min-width: 0;
    flex: 1;
    max-width: 28rem;
  }

  .mz-about input {
    width: 100%;
    padding: 0 var(--mz-space-1);
    border: 0;
    border-radius: var(--mz-radius-md);
    background: transparent;
  }

  .mz-about input:hover,
  .mz-about input:focus {
    background: var(--mz-color-accent-bg);
    outline: none;
  }

  .mz-name {
    font-weight: var(--mz-weight-semibold);
    line-height: var(--mz-leading-tight);
  }

  .mz-description {
    color: var(--mz-color-text-muted);
    font-size: var(--mz-text-xs);
  }

  .mz-actions {
    display: flex;
    align-items: center;
    gap: var(--mz-space-2);
    flex: none;
  }
</style>
