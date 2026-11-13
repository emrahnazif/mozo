/**
 * One editor's state, shared by its components.
 *
 * Made per editor and handed down through Svelte's context, not held at module level: two editors
 * on one page -- a host showing two workflows side by side -- would otherwise share one canvas's
 * file, one run's progress, one workflow's name.
 */
import { getContext, setContext } from 'svelte';
import { writable } from 'svelte/store';

const KEY = Symbol('mozo-editor');

/** A workflow that says nothing about itself: the name and description an empty canvas has. */
export const noAbout = () => ({ name: '', description: '' });

export function createState() {
    return {
        /** Every node the server offers, keyed by name -- see `fetchCatalogue` in api.js. */
        catalogue: writable({}),
        /** Whether the properties drawer is open. */
        drawerOpen: writable(false),
        /**
         * The file the next run reads, chosen on a node whose parameter is a `source`. An image or
         * a video, and this does not know which: deciding is the server's, from the extension.
         */
        chosenFile: writable(null),
        /**
         * A run in progress, or null between runs: `{items, seconds, preview, node, done}`. Not the
         * results -- a run over ten thousand items has none a canvas could hold.
         */
        running: writable(null),
        /** What the workflow says about itself: `{name, description}`, both optional. */
        about: writable(noAbout()),
    };
}

export const provideState = (state) => setContext(KEY, state);
export const useState = () => getContext(KEY);
