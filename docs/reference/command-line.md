# Command line reference

<!--
summary: The mozo command: start the HTTP server, run a workflow file over its whole source or one test item, and print the version.
label: Command line
keywords: mozo, command line, cli, mozo start, mozo run
-->

Installing mozo installs the `mozo` command. `mozo --help` lists the commands and
`mozo <command> --help` the options of one.

## mozo start

```bash
mozo start [--host HOST] [--port PORT] [--workers N] [--reload]
```

Start the HTTP server.

| Option | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Address to listen on. |
| `--port` | `8000` | Port to listen on. |
| `--workers` | `1` | Worker processes. Each loads its own copy of every model it serves, so memory grows with each one. |
| `--reload` | off | Restart when the code changes, for development. Forces one worker. |

The server reads `MOZO_ENABLE`, `MOZO_CACHE`, `MOZO_OFFLINE`, `MOZO_BASE_URL` and
`MOZO_PREPARED_MB` from the environment; [Configuration](/docs/deployment/configuration.md) has
them. [HTTP server](/docs/server.md) is the guide.

## mozo run

```bash
mozo run WORKFLOW [--file PATH] [--set NAME=VALUE ...] [--test]
```

Run a workflow from a JSON file, with no browser and no server. It runs over the whole source: every
image in a folder, every frame of a video. Progress is printed as it goes; nodes that write files
have written them when it ends.

| Option | Description |
|---|---|
| `--file` | Run on this image, video or folder instead of the one saved in the workflow. |
| `--set NAME=VALUE` | Override a parameter, by name or as `node.name` for one node's. Repeatable. |
| `--test` | Run one item and print what every node produced, instead of the whole source. |

```bash
mozo run blur_faces.json --file example.jpg --set threshold=0.5
```

```text
Blur faces -- Find every face and blur it.
  1 item in 2.8s (0/s)
```

`--test` describes each node's output in a line: an image's size, how many detections and of what,
a depth map's range. Use it to check a graph does its job before you point it at ten thousand
files:

```text
Blur faces -- Find every face and blur it.
  read: a 1920x1281 image
  find: 5 x ['face']
  blur: a 1920x1281 image
  save: nothing (an end)
```

[Run a workflow](/docs/workflows/python.md) covers the same from Python.

## mozo version

```bash
mozo version
```

```text
Mozo version 1.10.0
```
