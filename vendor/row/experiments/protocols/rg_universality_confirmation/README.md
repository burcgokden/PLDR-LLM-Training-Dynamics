# Transition-kernel RG and universality confirmation

This generated campaign is separate from the normal-stability confirmation
program. R0 through R3 test the projected Markov kernel, exact blocking,
closure defects, discrete RG flow, and continuous embeddability. U1 through
U3 test the Gaussian fixed point and transfer across three architecture
cells. Failure of R2 closure or R3 embeddability stops the campaign before
the longer universality tier.

The measured object is a branch-resolved lifted normal trajectory ensemble.
Direct b-step kernels are estimated independently and compared with iterated
one-step kernels. Hidden-stratum tests expose projection dependence.
Continuous flow is never inferred from a matrix logarithm unless the
principal real generator reconstructs the map and has positive stability
and nonnegative stationary diffusion edges.

The machine allocation is two RTX 4090 GPUs with a 20 GiB allocation cap on
each device. The full program has a hard limit of 128 GPU-hours and 96
wall-clock hours. It is not required for completion of the shorter
normal-stability campaign.
