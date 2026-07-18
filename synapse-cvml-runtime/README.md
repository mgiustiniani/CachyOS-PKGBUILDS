# Synapse Ryzen AI CVML integration

License-gated importer and diagnostics for AMD's Ryzen AI CVML C++ perception
library. CVML supplies face detection, face mesh, and depth estimation and can
explicitly request its NPU backend. It is separate from ComfyUI diffusion,
which runs on Radeon ROCm.

The AMD CVML binaries are not included in this package, pacman repository, or
ISO. Review `Ryzen-AI-CVML-Library/LICENSE.txt`, obtain the SDK from AMD, and
import it explicitly:

```sh
sudo synapse-cvml-import install --source /path/to/Ryzen-AI-CVML-Library \
  --accept-amd-cvml-eula --json
synapse-cvml doctor
sudo synapse-cvml build-samples
```

Do not install Ubuntu XRT packages over the Arch XRT stack. Imported libraries
remain side by side under `/opt/synapse/cvml`.
