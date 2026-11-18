# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tsubasasurf1222-arch/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/tsubasasurf1222-arch/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A launch-day Twitch viewership forecaster for Grand Theft Auto VI. The program reads one case file — everything known at the end of release day's first hour, including dozens of past big launches with their full release-day curves — and writes five quantiles of the viewer count for each of hours 2 to 24. Each past launch contributes its day shape relative to its own first hour; shapes are weighted by an exponential kernel on opening scale, with a weighted linear tilt for scale, so the launches that opened most like this one shape the forecast. Scored by mean pinball loss on the log scale; in leave-one-out backtests over the practice comparables it roughly halves the persistence baseline's loss (0.20 against 0.46). Pure Python standard library, deterministic, no network.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=14561xbg&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
