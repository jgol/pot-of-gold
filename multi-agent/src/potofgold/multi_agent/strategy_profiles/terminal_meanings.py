"""Terminal descriptions used by strategy_profiles.terminal_registry.

Each entry preserves the original (short prose, meaning, unit) tuple used to
explain a strategy's signals to the assessment models.
"""

# Values are (short description, meaning, unit).
TERMINALS = {
    "ATM_IV": ("ATM implied volatility", "at-the-money implied vol", "annualized vol"),
    "ATM_IV_5m": ("smoothed ATM IV", "ATM implied vol, 5-min bar", "annualized vol"),
    "VIXSpot": ("VIX level", "spot VIX", "vol points"),
    "VIXTermSlope": ("VIX term structure slope", "VIX term-structure slope", "vol points"),
    "VIXChange": ("overnight VIX change", "VIX change", "vol points"),
    "VIXMean5d": ("5-day mean VIX", "5-day mean VIX", "vol points"),
    "RealizedVol30m": ("30-min realized volatility", "30-min realized vol", "per-min vol"),
    "RealizedVol30m_5m": ("smoothed realized volatility", "30-min realized vol, 5-min bar", "per-min vol"),
    "RawSpread": ("ATM bid-ask spread", "bid-ask spread", "USD"),
    "RawSpread_5m": ("smoothed ATM spread", "bid-ask spread, 5-min bar", "USD"),
    "DeltaSpread1": ("1-bar spread change", "1-min change in bid-ask spread", "USD"),
    "DeltaSpread5": ("5-bar spread change", "5-min change in bid-ask spread", "USD"),
    "MinutesToClose": ("minutes to close", "minutes to the 16:15 ET PM settlement (not the 16:00 cash close)", "minutes"),
    "BarOfDay": ("bar of day", "minute-bar index within the session", "bars"),
    "SessionReturn": ("intraday return", "return since the open", "return"),
    "SessionPosition": ("session position (0=low, 1=high)", "fraction through the session", "0..1"),
    "OvernightGap": ("overnight gap", "overnight SPX gap, prev-close to open", "return"),
    "GridReliability": ("IV grid reliability", "option-grid data-quality score", "0..1"),
    "PutCallSkew": ("put-call skew", "put-minus-call 25-delta IV skew (risk-reversal)", "vol"),
    "PutCallSkew10": ("10-delta put-call skew", "put-minus-call 10-delta IV skew", "vol"),
    "ThetaUrgency": ("theta urgency (1/√time)", "time-decay urgency", "proxy"),
    "SPXReturn3d": ("3-day SPX return", "3-day SPX return", "return"),
    "RV5d": ("5-day realized volatility", "5-day realized vol", "annualized vol"),
    "IVRVGap5d": ("5-day IV-RV gap (variance premium)", "implied minus realized vol, 5-day", "vol"),
    "PredRV15": ("predicted 15-min RV", "predicted realized vol, 15-min", "vol"),
    "PredRV30": ("predicted 30-min RV", "predicted realized vol, 30-min", "vol"),
    "PredSpread": ("predicted spread change", "predicted spread", "USD"),
    "PredGammaAccel": ("predicted gamma acceleration", "predicted gamma acceleration", "proxy"),
    "PredSmileConvexity": ("predicted smile convexity change", "predicted smile convexity", "proxy"),
    "PredJump": ("predicted jump probability", "predicted jump", "proxy"),
    "PredFlowToxicity": ("predicted flow toxicity", "model-predicted order-flow toxicity", "score"),
    "NetGammaProxy": ("net-gamma proxy", "dealer net-gamma proxy", "proxy"),
    "CalDayOfWeek": ("day of week", "day of week", "0=Mon..4=Fri"),
    "CalDaysToFomc": ("days to FOMC", "signed calendar days to the nearest FOMC (negative = after the meeting)", "days"),
    "CalIsOpexWeek": ("OPEX week", "monthly-OPEX week flag", "0/1"),
    "CalTurnOfMonth": ("turn of month", "turn-of-month flag", "0/1"),
    "RegimeAboveLow": ("regime above low-vol", "vol-regime: above-low flag (L1 probe)", "prob"),
    "RegimeIsHigh": ("regime is high-vol", "vol-regime: high flag (L1 probe)", "prob"),
    "RegimeIsPremium": ("regime is crisis", "vol-regime: premium flag (L1 probe)", "prob"),
}

# Position variables use runtime units; Delta is a temporal operator.
POSITION_STATE = {
    "UnrealizedProfitPct": ("unrealized P/L %", "current unrealized P/L", "% of max"),
    "ProfitGiveback": ("profit giveback", "unrealized profit given back from the peak", "% of peak"),
    "MaxAdverseExcursion": ("max adverse excursion", "worst adverse move since entry", "%"),
    "MaxFavorableExcursion": ("max favorable excursion", "best favorable move since entry", "%"),
}

# Categorical values are (short description, meaning).
REGIME_LITERALS = {
    "LOW_VOL": ("low vol", "low-vol regime"),
    "MID_VOL": ("mid vol", "mid-vol regime"),
    "HIGH_VOL_STRESSED": ("high vol stressed", "high-vol stressed regime"),
    "HIGH_VOL_PREMIUM": ("high vol premium", "high-vol premium regime"),
}
SIDE_LITERALS = {
    "CALL": ("call", "call side"),
    "PUT": ("put", "put side"),
    "NEUTRAL": ("neutral", "neutral (no side)"),
}
