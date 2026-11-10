# Library architecture

The public workspace contains Inferno, its curated internal PandoraPurple
component crate, dk-tensor, dk-autograd and the native dk-cuda-ffi kernel adapter.
It contains no model application binary, application trainer or bridge objective.

The tensor API has two explicit families. Root Tensor/Variable use dk-tensor and
the generalised autograd engine. Candle/nn and core VAE/legacy GP use Candle from
PR #4044. `candle_apply` connects Candle computations to dk-autograd with a
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

The existing root numerical/operator APIs and the internal component crate have
separate operator/GP traits. Use one family consistently. Both use the exact same
pinned Candle tensor type.
