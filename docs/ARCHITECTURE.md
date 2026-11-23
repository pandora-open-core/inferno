# Library architecture

The inferno-rs package contains the tensor, autograd, pandora, cuda and pudding
modules. The pandorapurple public alias preserves existing model import paths.
It contains no model application binary, application trainer or bridge objective.

The tensor API has two explicit families. Root Tensor/Variable use dk-tensor and
the generalised autograd engine. Candle/nn and core VAE/legacy GP use Candle from
the crates.io registry. `candle_apply` connects Candle computations to dk-autograd with a
first-order CPU VJP node. There is no implicit tensor conversion.

VAE components expose forward operations and caller-owned parameters. Noise,
concentration, Gaussian algebra, latent SSM, EWMA and time embeddings are direct
VAE support. Legacy GP predictors retain historical feature layout conventions.
FlowMatchingModel supplies an independent velocity-field model option.

Stochastics contains GmmSchrodingerBridge and LangevinSDE, plus their mixture,
configuration and direction value types. Flows contains latent GEOC reference
geometry and barycentric drift, and Anderson time reversal. Kernel Sinkhorn
iterations compute component couplings with gradients. Bridge couplings can
weight decoder reconstruction and Dirichlet mean-probability divergence losses.

The existing root numerical/operator APIs and the internal model module have
separate operator/GP traits. Use one family consistently. Both use the exact same
pinned Candle tensor type.

Registry Candle 0.11.1 does not include the historical convolution patches.
The model module uses `conv_t::conv1d_diff`, a differentiable gather/matmul
implementation, for registered Conv1d layers and Mamba depthwise kernels.
Transposed convolution uses the same path. This preserves input and weight
gradients for short padded inputs and non-contiguous weights without a Candle
fork. The gather materializes convolution patches and may use more memory than
native convolution. The public `inferno::convolution` module exposes the 1D and 2D gather/matmul
paths for custom components, including grouped and dilated convolutions.
Direct calls to re-exported Candle APIs retain upstream
behavior; these safeguards apply to Inferno model components.
