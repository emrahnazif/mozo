<script>
  /**
   * One node on the canvas: its kind, a line about it, and a handle per port.
   *
   * Inputs along the top and outputs along the bottom, each coloured by the type it carries (see
   * `.mz-port-*` in editor.css). Every input is required: a node whose input carried a default is
   * refused when it registers, because an input arrives over a connection and a default is a value
   * nothing can produce.
   */
  import { Handle, Position, useEdges } from '@xyflow/svelte';
  import CategoryIcon from './CategoryIcon.svelte';
  import { useState } from './state.js';

  export let data;
  export let id;

  const edges = useEdges();
  const { catalogue } = useState();

  $: info = $catalogue[data.nodeType] || {};
  $: inputs = info.inputs || [];
  $: outputs = info.outputs || [];
  $: fed = new Set($edges.filter(edge => edge.target === id).map(edge => edge.targetHandle));
  $: feeding = new Set($edges.filter(edge => edge.source === id).map(edge => edge.sourceHandle));
  $: set = Object.keys(data.parameters || {}).length;

  /** Where along the edge a port's handle sits: spread evenly, one port centred. */
  const along = (index, count) => `left: ${(100 / (count + 1)) * (index + 1)}%;`;
</script>

{#each inputs as port, index (port.name)}
  <Handle type="target" position={Position.Top} id={port.name}
          class="mz-port-{port.type} {fed.has(port.name) ? 'mz-connected' : ''}"
          style={along(index, inputs.length)} />
{/each}

<div class="mz-entry">
  <span class="mz-entry-icon"><CategoryIcon category={info.category} /></span>
  <div>
    <p class="mz-title">{data.nodeType}</p>
    <p class="mz-entry-about">
      {set ? `${set} parameter${set === 1 ? '' : 's'} set` : info.description || ''}
    </p>
  </div>
</div>

{#each outputs as port, index (port.name)}
  <Handle type="source" position={Position.Bottom} id={port.name}
          class="mz-port-{port.type} {feeding.has(port.name) ? 'mz-connected' : ''}"
          style={along(index, outputs.length)} />
{/each}
