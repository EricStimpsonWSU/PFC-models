"""Phase-diagram machinery that does not depend on the model.

Substrate, in the catalogue's sense: nothing here changes when the physics
changes. Every model-specific piece - field names, builders, the structures a
model can form and their ansatz, the cell geometry, acceptance diagnostics, and
the analytic uniform (liquid) state - lives in the model's own adapter module,
`pfc_models.models.<slug>.phase_diagram`, which every supporting model provides
with the same names (a documented contract, not a base class).

Modules:

* `q0star` - strain-free lattice scale q0* and fmin(phibar) for one structure at
  one temperature: relaxation by time stepping, a q scan with spline refinement,
  and `scan_density` for one batch task (one structure, one temperature, one
  density range).
* `common_tangent` - coexistence between two free-energy branches of one conserved
  density (common tangent / Maxwell construction), the lower-envelope check over
  all branches at one temperature, and Maxwell-line assembly over temperatures.
  Models with several conserved densities need a tangent-plane construction in a
  sibling module.
"""
