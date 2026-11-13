<script>
    /**
     * A Heroicon, outline at 24px, inlined at build time from the `heroicons` package -- the same
     * set babasu uses. MIT licensed; the licence ships beside the built editor as
     * `heroicons-LICENSE`.
     */
    import play from 'heroicons/24/outline/play.svg?raw';
    import forward from 'heroicons/24/outline/forward.svg?raw';
    import stop from 'heroicons/24/outline/stop.svg?raw';
    import arrowDownTray from 'heroicons/24/outline/arrow-down-tray.svg?raw';
    import arrowUpTray from 'heroicons/24/outline/arrow-up-tray.svg?raw';
    import magnifyingGlass from 'heroicons/24/outline/magnifying-glass.svg?raw';
    import xMark from 'heroicons/24/outline/x-mark.svg?raw';

    const ICONS = {
        play, forward, stop, 'arrow-down-tray': arrowDownTray, 'arrow-up-tray': arrowUpTray,
        'magnifying-glass': magnifyingGlass, 'x-mark': xMark,
    };

    export let name;

    // The package's own markup, with the class every icon here is sized by.
    $: svg = ICONS[name].replace('<svg ', '<svg class="mz-icon" ');
</script>

{@html svg}
