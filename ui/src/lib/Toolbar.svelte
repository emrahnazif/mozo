<script>
  /**
   * What you do to a workflow as a whole: test it, run it, save it, open one.
   *
   * **Two verbs, because they are two different things.** Test puts one item through the graph and
   * draws every node's output, which is what you want while wiring. Run puts the whole source
   * through and draws none of them -- see `process` in api.js for why it cannot.
   *
   * The file a run reads is not here. It belongs to the node that reads it, where the person is
   * already looking when they wonder what it will run on -- see the `source` widget in
   * PropertiesPanel.
   *
   * The workflow's name and description sit at the start of the bar: they belong to the whole
   * workflow, like these verbs, and they are what someone opening the file later sees first.
   */
  import { about } from './stores.js';

  export let testWorkflow;
  export let runWorkflow;
  export let cancelRun;
  export let exportWorkflow;
  export let importWorkflow;
  export let isExecuting;
  export let isRunning;

  let fileInput;
</script>

<nav class="navbar is-fixed-top border-bottom" aria-label="main navigation">
  <div class="navbar-menu">
    <div class="navbar-start">
      <div class="navbar-item workflow-about">
        <input class="input is-static workflow-name" type="text" placeholder="Untitled workflow"
               aria-label="Workflow name" bind:value={$about.name} />
        <input class="input is-static is-small workflow-description" type="text"
               placeholder="What it does" aria-label="Workflow description"
               bind:value={$about.description} />
      </div>
    </div>
    <div class="navbar-end">
      <div class="navbar-item">
        <div class="buttons">
          <button class="button" on:click={testWorkflow}
                  disabled={isExecuting || isRunning}
                  title="One item through the graph, with every node's output shown">
            {isExecuting ? 'Testing...' : 'Test'}
          </button>

          {#if isRunning}
            <button class="button is-danger is-light" on:click={cancelRun}>Cancel</button>
          {:else}
            <button class="button is-dark" on:click={runWorkflow} disabled={isExecuting}
                    title="Every item the input produces">
              Run
            </button>
          {/if}

          <button class="button is-dark" on:click={exportWorkflow}>Export</button>
          <button class="button" on:click={() => fileInput.click()}>Import</button>
        </div>
      </div>
    </div>
  </div>

  <input
    bind:this={fileInput}
    type="file"
    accept=".json"
    class="is-hidden"
    on:change={importWorkflow}
  />
</nav>

<style>
  .workflow-about {
    flex-direction: column;
    align-items: flex-start;
    gap: 0;
    min-width: 18rem;
  }
  .workflow-name {
    font-weight: 600;
    height: 1.8em;
    padding: 0 0.25rem;
  }
  .workflow-description {
    height: 1.6em;
    padding: 0 0.25rem;
    color: var(--bulma-text-weak, #7a7a7a);
  }
  .workflow-about input:hover,
  .workflow-about input:focus {
    background: var(--bulma-scheme-main-bis, #fafafa);
  }
</style>
