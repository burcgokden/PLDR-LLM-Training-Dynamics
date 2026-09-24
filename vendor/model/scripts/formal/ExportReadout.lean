import ModelRG
import Lean
open Lean Elab Command

elab "#export_modelrg " n:ident : command => do
  let info ← getConstInfo n.getId
  let row := Json.mkObj [("name", toJson n.getId.toString),
    ("type", toJson (reprStr info.type)),
    ("pretty_type", toJson (← liftTermElabM <| PrettyPrinter.ppExpr info.type).pretty)]
  logInfo m!"STATEMENT_JSON {row.compress}"

#export_modelrg ModelRG.readout_squared_norm_error_bound
#export_modelrg ModelRG.readout_relative_scale_budget
#export_modelrg ModelRG.readout_combined_law_budget
