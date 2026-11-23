import Pudding.Extensions
import Lean.Util.CollectAxioms

open Lean in
run_cmd do
  let env ← getEnv
  let mut checked := 0
  let mut compilerInternals := 0
  let mut visited : CollectAxioms.State := {}
  -- Inspect imported names first, fetching ConstantInfo only for this library.
  -- Materialising every imported ConstantInfo needlessly pages all of mathlib.
  let mut roots : NameSet := {}
  for data in env.header.moduleData do
    for name in data.constNames do
      if (`Pudding).isPrefixOf name then roots := roots.insert name
  -- Include any declarations added in this audit module as well.
  let localNames := env.constants.foldStage2 (fun names name _ => names.push name) #[]
  for name in localNames do
    if (`Pudding).isPrefixOf name then roots := roots.insert name
  for name in roots.toList do
    if let some info := env.find? name then
      if name.isInternalDetail && !info.isTheorem then
        compilerInternals := compilerInternals + 1
        continue
      let (_, next) := ((CollectAxioms.collect name).run env).run visited
      visited := next
      checked := checked + 1
  let rejected := visited.axioms.filter fun ax =>
    ax != ``propext && ax != ``Classical.choice && ax != ``Quot.sound
  unless rejected.isEmpty do
    throwError "unapproved transitive axioms {rejected}"
  if checked == 0 then throwError "empty proof closure"
  logInfo m!"Audited {checked} declarations, including transitive imported axioms; {compilerInternals} internal non-theorem compiler declarations excluded as roots"

#print axioms Pudding.Transport.singleton_error_converges
#print axioms Pudding.Transport.row_residual_identity
#print axioms Pudding.Symbolic.code_packet_lane
