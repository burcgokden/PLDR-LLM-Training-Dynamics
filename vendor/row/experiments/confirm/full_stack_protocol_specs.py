"""Primary protocol wording for the complete-stack confirmation chain."""

from __future__ import annotations


def apply(specs):
    specs["E0"].update({
        "title": "Disjoint physical row domain and independent cover remainder",
        "target": (
            "A physical row domain constructed only from its frozen construction "
            "partition, exact membership tests on disjoint validation rows, and "
            "an independent derivative-modulus times cell-radius remainder."
        ),
        "estimand": (
            "Exact validation membership, full-matrix direct upper bounds, "
            "independent cover remainders, and strict cover utility margins."
        ),
        "procedure": [
            "Freeze construction and validation row identifiers before evaluating either partition.",
            "Build each chordal cover from construction rows only.",
            "Reconstruct every matrix coordinate at every registered layer and cell.",
            "Bound each cover remainder by the validated second-derivative modulus times its exact midpoint radius.",
            "Test validation rows against the frozen rational box union without changing the cover.",
            "Require the cover remainder plus reserved tail budget to lie strictly below the physical criterion.",
        ],
        "decision": (
            "CONFIRMED requires every validation row to belong to the frozen "
            "domain, every held-out full Jacobian to remain below its certificate, "
            "all derivative checks to close, and every cover utility margin to be positive."
        ),
        "prediction": "The construction-only physical domain certifies the direct row-map tube with positive utility margin.",
    })
    specs["E0"]["thresholds"].update({
        "validation_domain_membership_indicator_min": 1.0,
        "cover_utility_margin_min": 0.0,
    })

    specs["E1"].update({
        "title": "Actual program-state successor and operation-ordered transport",
        "target": (
            "The exact successor on parameters, both Adam moments, optimizer "
            "clock, scheduler phase, learning rate, row-cover state, and intervention state."
        ),
        "procedure": [
            "Declare rational source and target cells for every program-state coordinate.",
            "Split any cell that touches a clipping boundary or square-root branch point.",
            "Evaluate clipped AdamW in implemented order with exact interval derivatives.",
            "Check the complete image against the declared successor cell.",
            "Derive the positive state-radius map from the actual center image, derivatives, exogenous radii, and interval remainder.",
            "Retain the matrix-valued lifted recurrence as a separate exact identity.",
        ],
        "decision": (
            "CONFIRMED requires exact program-state image inclusion and the "
            "operation-ordered transport identities on every registered edge."
        ),
        "prediction": "Every accepted program cell maps into its registered successor with no caller-selected radius multiplier.",
    })

    specs["E2"].update({
        "title": "Complete Jacobian stack and derivative-defined normal closure",
        "target": (
            "The direct sum of every entry of every registered row-map Jacobian, "
            "together with a normal closure constructed from derivatives rather than a trajectory fit."
        ),
        "estimand": (
            "Registry completeness, right-inverse residual, closure-complement "
            "norm, frozen forcing, and validation residual envelopes through second order."
        ),
        "procedure": [
            "Enumerate every layer from zero through architecture depth minus one and every registered cover cell.",
            "Stack all full Jacobian entries in the frozen registry order.",
            "Compute B as the parameter derivative of the stack and DQ as the parameter derivative of normal velocity.",
            "Freeze a right inverse R, verify the norm of BR minus I is below one, and define K as DQ times R.",
            "Charge separate runtime errors to the right-inverse residual, closure complement, validation-center residual, and validation-derivative residual.",
            "Freeze Q-bar minus K Psi-bar once on the construction partition.",
            "Validate center, first-derivative, and second-derivative closure residuals without refitting.",
        ],
        "decision": (
            "CONFIRMED requires complete registry coverage, right-inverse residual "
            "below one, and every frozen normal-closure residual within its primitive envelope."
        ),
        "prediction": "The derivative-defined dense normal operator closes the complete stack with an explicit complement defect.",
    })

    specs["E3"].update({
        "title": "Complete block comparison with every cross coupling",
        "target": (
            "A nonnegative comparison matrix derived from all diagonal and "
            "off-diagonal blocks of the lifted stack and program-state operator."
        ),
        "procedure": [
            "Partition the complete operator into a contiguous direct sum of all registered blocks.",
            "Supply source and target quadratic metrics for every block.",
            "Charge outward runtime conversion error to every ordered block pair.",
            "Compute every source-to-target metric gain, including cross-layer and stack-to-state couplings.",
            "Compose positive affine edge maps in temporal order over every admissible length-four path.",
            "Take the entrywise path envelope and test a positive weighted witness when infinite continuation is claimed.",
        ],
        "decision": (
            "CONFIRMED requires all ordered blocks, all registered paths, all "
            "runtime charges, and the claimed finite or infinite positive comparison witness."
        ),
        "prediction": "No stable scalar projection can hide an orthogonal or cross-layer expanding mode.",
    })

    specs["E4"].update({
        "title": "Primitive source ledger and runtime-to-exact bridge",
        "target": (
            "Source bounds derived from primitive intervals and derivatives, "
            "with explicit runtime error charges and no geometric trace fit."
        ),
        "procedure": [
            "Enclose every runtime primitive by nextafter-outward exact binary rationals.",
            "Record kernel, reduction, reconstruction, and serialization error sources.",
            "Derive tangent, transient, persistent, and finite-impulse bounds from the program cell and normal residual ledger.",
            "Validate source derivatives on a disjoint partition without changing the frozen forcing.",
            "Reject terminal-value substitution, fitted rates, and zero nonlinear bounds without an exact affine proof.",
        ],
        "decision": (
            "CONFIRMED requires every forcing coordinate to have primitive "
            "provenance and every runtime-to-exact charge to be included in the accepted operator."
        ),
        "prediction": "Primitive source bounds close without extrapolating a fitted disturbance trace.",
    })

    specs["E5"].update({
        "title": "Derived successor-cell self-map",
        "target": (
            "Forward invariance of the actual program and complete lifted stack "
            "under radii derived internally from interval images and derivatives."
        ),
        "procedure": [
            "Evaluate each actual program cell and its complete lifted operator.",
            "Use target-cell centers fixed before the image is opened.",
            "Derive state and source contributions coordinate by coordinate.",
            "Add interval remainders and complete forcing provenance.",
            "Verify source-to-target inclusion and every within-path prefix budget.",
            "Reject any caller-selected entry, successor, or auxiliary radius multiplier.",
        ],
        "decision": (
            "CONFIRMED requires exact image inclusion and nonnegative source, "
            "target, prefix, and closed-path margins on every registered edge."
        ),
        "prediction": "The invariant tube is a consequence of the implemented successor, not a fitted tube around a trace.",
    })

    specs["E6"].update({
        "title": "Finite, frozen, and controlled-tail collapse conclusions",
        "target": (
            "A conclusion whose temporal scope is exactly one of a finite "
            "implemented schedule, an evaluation-only frozen checkpoint, or a controlled infinite tail."
        ),
        "procedure": [
            "Declare one of the three allowed time semantics before evaluating outcomes.",
            "Require every registered target cell to own an outgoing successor before claiming continued training.",
            "For a finite schedule, compose only implemented registered successors and stop at the terminal update.",
            "For a frozen checkpoint, verify unchanged model and row-cover digests with optimizer updates disabled.",
            "For a controlled infinite tail, enumerate every successor path and verify P v below kappa v with kappa below one.",
            "Project the complete stack to every layer's direct operator bound and add its independent cover remainder.",
            "Compute strict common-layer entry only above the complete persistent floor.",
        ],
        "decision": (
            "CONFIRMED requires the declared time semantics to close, every direct "
            "bound to lie below the physical criterion, and a positive floor margin; "
            "only the controlled-tail mode may claim all-future training persistence."
        ),
        "prediction": "The certificate yields the strongest collapse statement justified by the implemented future dynamics.",
    })
    return specs
