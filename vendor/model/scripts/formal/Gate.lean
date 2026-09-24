import ModelRG
import Lean.Util.CollectAxioms
open Lean Elab Command

-- Namespace plus module ownership includes private and generated declarations.
elab "#audit_modelrg" : command => do
  let env ← getEnv
  let mut count : Nat := 0
  let mut theoremCount : Nat := 0
  let mut names : Array Name := #[]
  for i in [:env.header.moduleNames.size] do
    if (`ModelRG).isPrefixOf env.header.moduleNames[i]! then
      names := names ++ env.header.moduleData[i]!.constNames
  let locals := env.checked.get.constants.foldStage2 (fun ns n _ =>
    if (`ModelRG).isPrefixOf n then ns.push n else ns) (#[] : Array Name)
  names := (names ++ locals).toList.eraseDups.toArray
  for name in names do
    if let some info := env.find? name then
      count := count + 1
      match info with
      | .axiomInfo _ => throwError "Untrusted owned axiom: {name}"
      | .thmInfo _ => theoremCount := theoremCount + 1
      | _ => pure ()
      for ax in (← Lean.collectAxioms name) do
        unless ax == ``propext || ax == ``Classical.choice || ax == ``Quot.sound do
          throwError "Declaration {name} depends on untrusted axiom {ax}"
  if count == 0 then throwError "Empty declaration inventory"
  logInfo m!"AXIOM_GATE_PASS declarations={count} theorem_declarations={theoremCount}"

#audit_modelrg
