// The dev server's page: the editor, against a mozo server on :8000 (see `server.proxy` in
// vite.config.js).
import { mount } from './editor.js';

mount(document.getElementById('app'), { base: `${location.origin}/workflow/` });
