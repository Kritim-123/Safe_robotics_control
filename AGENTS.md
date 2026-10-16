# Safe Robotics Control agent workflow

This project develops a MuJoCo Go2 system with SNN-based robot skill selection.
The three reusable project agents are defined in `.codex/agents/`:

- `snn_researcher`: investigate LIF and alternative spiking neuron models,
  encoding, temporal decisions, and learning methods for choosing skills.
- `skill_developer`: expand and independently test the robot's skill library.
- `system_integrator`: incorporate the research and skills into the SNN-based
  control mechanism and validate the complete loop.

## Starting a task

Use one agent in one chat at a time, including substantive SNN/skill integration
milestones. This is the user's standing preference as of October 16, 2026.
Do not spawn or delegate to other agents or chats unless the user explicitly
changes this preference. The role descriptions above are reference checklists
for the single agent to apply sequentially.

Read `README.md`, `docs/project_status.md`, `docs/hybrid_system.md`, and
`docs/hybrid_validation.md` before planning. Inspect the current code and skill
inventory: selector actions, callable skills, and learned policies are different
things. Do not assume a fixed number of skills or treat historical results as
fresh test evidence.

1. Establish the requested milestone and measurable acceptance criteria.
2. Inspect research and skill requirements sequentially, then draft the
   observation/skill contract and validation plan in the current chat.
3. Agree on observations, units, skill identifiers, preconditions, termination,
   interruption, failure signals, and recovery. Assign file ownership before edits.
4. Complete research and skill development sequentially after that agreement.
   Use the research recommendation and tested skill interfaces before dependent
   integration work.
5. Test the integrated system. Return failures to their owners, then revalidate
   affected behavior. The parent independently reviews changes and evidence.
6. Report changes, actual commands and results, limitations, and remaining work.

## Ownership and evidence

The researcher owns its research brief; the developer owns assigned skill and
controller changes; the integrator owns assigned hybrid integration changes.
Assign shared tests, configurations, notebooks, and policy artifacts to one
writer at a time. Do not overwrite another agent's or the user's changes.
Role instructions do not provide separate sandbox permissions.

Use `docs/agents/` for milestone research briefs, skill catalogs, and integration
reports. Use `tmp/` for scratch work and the existing `output/` conventions for
evaluation artifacts. Run commands from the project root, and resolve artifact
paths explicitly to avoid writing into neighboring projects.

Use the existing project Python environment when available. On Windows,
`.venv-win/Scripts/python.exe -m unittest discover -s tests -v` runs the suite.
Start with relevant tests; run the broader suite for integrated controller
changes. Inspect documented evaluation commands before running experiments.
Report missing dependencies and unexecuted checks rather than asserting success.

Preserve the distinction between SNN skill selection and low-level joint PID.
Maintain actuator limits and existing guards. Test eligibility, fallback, and
handoffs explicitly. Keep learned-policy provenance and training/evaluation
splits reproducible. Simulation evidence does not certify physical robot safety
or establish that SNNs outperform other methods. Hardware deployment is outside
a simulation task's scope.

## Example invocation

> Use snn_researcher, skill_developer, and system_integrator for this milestone:
> [describe the desired capability]. Compare suitable LIF variants for skill
> selection, develop the missing skills, and integrate and validate the result.
> Agree on interfaces first, coordinate file ownership, and report reproducible
> evidence and unresolved failures.
