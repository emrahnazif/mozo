/**
 * The workflow editor, as something to put on a page:
 *
 *     import { mount } from "/workflow/assets/editor.js";
 *     const editor = mount(document.querySelector("#editor"), { document: saved });
 *
 * mozo's own page is this and nothing more (`_PAGE` in mozo/workflow/api.py). It mounts into a
 * shadow root, so its styles and the page's cannot reach each other; a host restyles it through
 * the `--mz-*` tokens, which do cross, set on the element it mounted into.
 */
import '@xyflow/svelte/dist/style.css';
import './styles/editor.css';
import App from './App.svelte';

const STYLESHEET = new URL(/* @vite-ignore */ 'editor.css', import.meta.url).href;

/**
 * Mount an editor into *element*, which it fills. Give the element a size.
 *
 * @param {HTMLElement} element Where to draw it.
 * @param {object} [options]
 * @param {object} [options.document] A workflow to open, in mozo's document format.
 * @param {boolean} [options.toolbar=true] Draw the toolbar. A host with buttons of its own turns
 *     it off and drives the editor through what this returns.
 * @param {string} [options.base] Where the workflow router is mounted, ending in `/`. Derived
 *     from this script's own address -- it is served at `<router>/assets/editor.js` -- so it is
 *     given only to reach a router somewhere else.
 * @param {HTMLElement} [options.panel] Where to draw the panel -- the node list, or the selected
 *     node's settings -- for a host that wants it in a column of its own. It fills the element,
 *     in a shadow root of its own, restyled by the same `--mz-*` tokens set on that element.
 *     Without it, the panel sits at the right of the editor's box.
 * @returns {{getDocument: function(): object, setDocument: function(object): void,
 *            test: function(File=): Promise<object>, on: function(string, function): function,
 *            destroy: function(): void}}
 */
export function mount(element, { document = null, toolbar = true, base, panel } = {}) {
    const target = window.document.createElement('div');
    target.style.height = '100%';
    const root = into(element, target);
    // The editor's styles are all scoped under `.mz-editor`, so the panel is drawn inside one;
    // and into a shadow root with the stylesheet, like the editor itself, so the page's styles
    // and the panel's cannot reach each other.
    const panelTarget = panel ? Object.assign(window.document.createElement('div'), { className: 'mz-editor' }) : null;
    const panelRoot = panel ? into(panel, panelTarget) : null;

    const app = new App({
        target,
        props: {
            base: base || new URL(/* @vite-ignore */ '..', import.meta.url).href,
            toolbar, document, panel: panelTarget,
        },
    });

    return {
        /** The workflow on the canvas, in mozo's document format. */
        getDocument: () => app.getDocument(),
        /** Open a workflow in place of the one on the canvas. Not reported as a change. */
        setDocument: (opened) => app.setDocument(opened),
        /** Test on *file*, or on the one chosen on the input node: `{success, results, error}`. */
        test: (file) => app.test(file),
        /**
         * Listen for *event*: `"change"`, with the document, whenever the person changes the
         * workflow. Returns a function that stops listening.
         */
        on: (event, listener) => app.$on(event, (e) => listener(e.detail)),
        /** Take the editor off the page. */
        destroy: () => {
            app.$destroy();
            root.replaceChildren();
            panelRoot?.replaceChildren();
        },
    };
}

/**
 * Draw *target* into *host*, in a shadow root holding the stylesheet; returns what holds it.
 * Held back until the stylesheet has arrived, so it is not seen unstyled for a moment. Hidden
 * rather than not drawn: the canvas measures itself as it mounts.
 */
function into(host, target) {
    if (import.meta.env.DEV) {
        // The dev server puts the styles in the page's head, where a shadow root would not see
        // them -- so in development only, it mounts straight into the element.
        host.classList.add('mz-host');
        host.replaceChildren(target);
        return host;
    }
    const root = host.shadowRoot || host.attachShadow({ mode: 'open' });
    const link = Object.assign(window.document.createElement('link'), { rel: 'stylesheet', href: STYLESHEET });
    target.style.visibility = 'hidden';
    link.onload = link.onerror = () => { target.style.visibility = ''; };
    root.replaceChildren(link, target);
    return root;
}
