# Interactive SNN decision inspector

Start from the project directory in WSL/Bash:

```bash
./.venv-win/Scripts/python.exe -m hybrid.inspector --run output/hybrid/latest
```

PowerShell:

```powershell
.\.venv-win\Scripts\python.exe -m hybrid.inspector --run output/hybrid/latest
```

Open http://127.0.0.1:8765 in your browser. Keep the terminal running; Ctrl+C
stops the server. Use `--port 8766` if that port is occupied, or `--no-browser`
to suppress automatic opening. Only Python and NumPy are required; there are
no frontend packages or external CDN assets. The server listens on loopback.

## What to try

1. Select the first saved encounter. The original choice is bypass left.
2. Change **left path clearance** from about 0.280 to **0.199 m**. The choice
   becomes bypass right, even though both paths are still allowed. This shows
   a learned score change, rather than a mask threshold crossing.
3. Select the second encounter, at 20.8 seconds. The original choice is wait.
4. Set **predicted wait clearance** to **0.010 m**. Wait still has the highest
   neural score, but is masked, so the supervisor chooses back off.
5. Choose **Hold recorded mask fixed**. The same edited inputs and neural
   scores now choose wait, isolating the effect of the mask.
6. Reset observations. Select a neuron and use the step slider or Play decision
   to inspect the 16-step spike train and membrane reset.

These examples refer to the saved `late_crossing` seed-0 run. Other runs can
have different features and reactions. Any SNN run with summary/events files
can be loaded, provided its model SHA-256 matches `--model`. A run with no
SNN decisions is rejected with an explanatory error.

## What each panel means

- **Observations:** all 12 real features, units, definitions, editable values,
  recorded values and clipped normalized values.
- **Decision:** four current scores, changes from recorded scores, mask state,
  raw network preference and final permitted action.
- **Sensitivity:** lower and raise each current feature by 0.25 training
  standard deviations independently, showing four score changes and final
  choices. Radius is bounded above zero and goal distance at zero. The exact
  tested endpoints are shown. Rows rank maximum absolute score response.
- **Neurons:** binary spikes across both layers and the selected neuron's
  membrane before and after the soft reset. The step slider highlights the
  current step; the full membrane trajectory remains visible for context.

This calls the production NumPy model for every experiment, including the
same normalization, binary firing and readout. It does not substitute a
JavaScript approximation. Model hash and replayed scores are verified when
loading the run. Five focused tests check replay parity, sensitivity against
independent production calls, fixed versus recomputed masks, all-blocked
behavior, invalid inputs, model mismatch and the HTTP endpoints. Browser checks verified editing,
decision switching, mask freezing, reset, neuron selection and step controls.

## Limits of the explanation

This is a saved-decision explorer, not a live robot controller. It neither
trains the network nor actuates the robot. Input experiments measure responses
of this model, not physical causality. Editing a derived clearance without
updating positions can describe an impossible scene. Recomputed mask means
applying the exact eligibility thresholds to the edited clearance values;
it does not rerun the geometric planner. The SNN's internal steps are not
seconds; membrane values are dimensionless. Sensitivity entries are finite
interventions, not probabilities, percentages of responsibility, or additive
feature attributions. The 0.25-standard-deviation interval can change rankings;
zero response can mean no firing threshold crossed or normalization clipped.
