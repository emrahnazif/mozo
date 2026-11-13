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
 * @returns {{getDocument: function(): object, setDocument: function(object): void,
 *            test: function(File=): Promise<object>, on: function(string, function): function,
 *            destroy: function(): void}}
 */
export function mount(element, { document = null, toolbar = true, base } = {}) {
    const target = window.document.createElement('div');
    target.style.height = '100%';
    let root;

    if (import.meta.env.DEV) {
        // The dev server puts the styles in the page's head, where a shadow root would not see
        // them -- so in development only, the editor mounts straight into the element.
        root = element;
        element.classList.add('mz-host');
        root.replaceChildren(target);
    } else {
        root = element.shadowRoot || element.attachShadow({ mode: 'open' });
        // Held back until the stylesheet has arrived, so the editor is not seen unstyled for a
        // moment. Hidden rather than not drawn: the canvas measures itself as it mounts.
        const link = Object.assign(window.document.createElement('link'),
                                   { rel: 'stylesheet', href: STYLESHEET });
        target.style.visibility = 'hidden';
        link.onload = link.onerror = () => { target.style.visibility = ''; };
        root.replaceChildren(link, target);
    }

    const app = new App({
        target,
        props: { base: base || new URL(/* @vite-ignore */ '..', import.meta.url).href, toolbar, document },
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
        },
    };
}
