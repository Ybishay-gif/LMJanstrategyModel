# Insurance Growth & Profitability Navigator

This app is tailored for your auto insurance acquisition workflow.

It combines:
- State strategy buckets
- State and state-segment profitability KPIs
- Channel-group bid elasticity tests
- Channel-group x state performance

to recommend how aggressively to scale each channel-group/state pair, including:
- Suggested bid adjustment
- Expected additional clicks
- Expected additional cost

## Inputs expected
Default paths are prefilled in the app sidebar:
- `/Users/YossiBen_Y/Desktop/State strategy`
- `/Users/YossiBen_Y/Downloads/State Data Jan 2026.csv`
- `/Users/YossiBen_Y/Downloads/State-Seg data 2026.csv`
- `/Users/YossiBen_Y/Downloads/Channel group data jan 2026.csv`
- `/Users/YossiBen_Y/Downloads/Channel group price exploration data.csv`
- `/Users/YossiBen_Y/Downloads/Channel group and state Jan 2026.csv`

## Run

```bash
cd "/Users/YossiBen_Y/Desktop/Lm model Jan 2026"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py
```

## Current scoring logic
- `Growth Score` favors click lift + win-rate lift and penalizes CPC lift.
- Best channel-group adjustment is selected under a max CPC increase constraint.
- Per channel-group/state recommendation blends:
  - growth score
  - local profitability (state-segment when bids >= threshold, else state)
  - state strategy aggressiveness bucket

All key thresholds/weights are adjustable in the sidebar.
# LMJanstrategyModel
# LMJanstrategyModel
