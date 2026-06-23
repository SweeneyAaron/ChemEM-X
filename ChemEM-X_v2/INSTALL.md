# Installing the ChemEM-X ChimeraX plugin

This plugin is a UI for the **ChemEM backend**. It does its work by running the
`chemem` command-line program as a local subprocess — there is no server or
network involved. So you need two things: the backend, then this plugin. No
GitHub account or `git` is required — download the code as ZIP files.

## 1. Install the backend first

Download the backend (repo `chemem2-dev`, branch `feature/mapq_score`) as a ZIP:

> https://github.com/SweeneyAaron/chemem2-dev/archive/refs/heads/feature/mapq_score.zip

Unzip it, then in **Terminal**:

```bash
cd ~/Downloads/chemem2-dev-feature-mapq_score
./install_macos_arm64.sh
```

Full backend walkthrough and troubleshooting:
[COLLABORATOR_SETUP.md](https://github.com/SweeneyAaron/chemem2-dev/blob/feature/mapq_score/COLLABORATOR_SETUP.md).

## 2. Install this plugin

Download this plugin (repo `ChemEM-X`, branch `folder-branch`) as a ZIP:

> https://github.com/SweeneyAaron/ChemEM-X/archive/refs/heads/folder-branch.zip

Unzip it. The bundle is the `ChemEM-X_v2` folder inside it (it contains
`bundle_info.xml`). From **inside ChimeraX**, install it from source:

```
devel install ~/Downloads/ChemEM-X-folder-branch/ChemEM-X_v2
```

(Adjust the path to wherever you unzipped it.) This pulls the bundle's
`scikit-spatial==7.0` dependency into ChimeraX's Python. Restart ChimeraX, then
open **Tools → Structure Prediction → ChemEM**.

## 3. Point the plugin at the backend

Easiest — launch ChimeraX from the activated backend env so it's auto-detected:

```bash
conda activate chemem
open -na ChimeraX
```

If the backend isn't auto-detected (e.g. conda installed in a non-standard
location), paste the executable path into the plugin's backend selector — the
installer prints it, typically `…/envs/chemem/bin/chemem`.

See [COLLABORATOR_SETUP.md](https://github.com/SweeneyAaron/chemem2-dev/blob/feature/mapq_score/COLLABORATOR_SETUP.md)
for the full walkthrough and troubleshooting.
