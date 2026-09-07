"""Performance and risk statistics computed from a `BacktestResult`.

Sharpe (gross and net), Newey-West t-stats at a lag equal to the holding
period, max drawdown, annualised volatility, turnover, and the deflated
Sharpe ratio (Bailey-Lopez de Prado, using registry `trial_count` — never a
function argument, DESIGN §8 G4) that `gates.py` consumes. Also the
sub-period and capacity-rescaled statistics gates G6 and G8 need, and the
inputs `report.py` renders into the IC pack's verdict table and charts.
"""
