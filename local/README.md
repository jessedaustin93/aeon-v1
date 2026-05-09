# Local Configuration

`local/` is for machine-specific configuration that should not be committed.

Use this directory when the portable defaults are not enough for your machine.

## Launcher Config

Expected files:

```text
local/launcher_config.example.json  committed portable example
local/launcher_config.json          ignored local override
```

Copy the example when you need local overrides:

```bash
cp local/launcher_config.example.json local/launcher_config.json
```

On Windows PowerShell:

```powershell
Copy-Item local\launcher_config.example.json local\launcher_config.json
```

## What Belongs Here

- local LM Studio executable path,
- local Obsidian executable path,
- local dashboard port override,
- local runner poll settings,
- local vault open command or vault name.

## What Does Not Belong Here

- source code,
- general documentation,
- real memory records,
- API keys,
- secrets,
- anything another machine should need from GitHub.

## Why This Exists

Aeon should be plug-and-play from the repo, but app launch paths are different on every machine. This folder keeps those paths out of shared source files while still allowing one-click local launch.
