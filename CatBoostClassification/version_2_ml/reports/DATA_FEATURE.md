# Dataset Column Guide

---

## ELO Rating Dataset

| Column          | What it is                                     | How to read the value                                                                                                        | Example (Spain) |
| --------------- | ---------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `team_name`     | Full country name                              | Text identifier                                                                                                              | Spain           |
| `country_code`  | Short country code                             | Used to merge with FIFA data                                                                                                 | ES              |
| `previous_rank` | ELO rank from last update                      | Lower = better. Compare to current rank to see if rising or falling                                                          | 1               |
| `elo`           | Current ELO score — the main strength number   | Higher = stronger. World class teams sit between 1900–2200. Below 1500 = weak team                                           | 2129            |
| `max_elo`       | Highest ELO score the team has ever reached    | Compare to current ELO — if close to max, team is at their historical best. Big gap = team is declining from their peak      | 2189            |
| `matches`       | Total international matches ever played        | Higher = more experienced. Use to normalise other stats (wins, goals) into rates                                             | 783             |
| `wins`          | Total wins all time                            | Don't use raw — divide by matches to get win rate                                                                            | 462             |
| `draws`         | Total draws all time                           | Don't use raw — divide by matches to get draw rate                                                                           | 183             |
| `losses`        | Total losses all time                          | Don't use raw — divide by matches to get loss rate                                                                           | 140             |
| `home_matches`  | Total matches played at home                   | Use as denominator for home win rate                                                                                         | 341             |
| `home_wins`     | Total home wins                                | Divide by home_matches to get home win rate                                                                                  | 138             |
| `home_draws`    | Total home draws                               | Optional — use if you want home draw rate                                                                                    | 302             |
| `goals_for`     | Total goals scored all time                    | Divide by matches to get attack strength per game                                                                            | 1595            |
| `goals_against` | Total goals conceded all time                  | Divide by matches to get defense strength per game. Lower = better defense                                                   | 699             |
| `recent_form`   | Sum of ELO point changes across last 6 matches | **Positive = good form**, winning against strong opponents. **Negative = bad form**, losing recently. Near 0 = mixed results | 178             |

---

## FIFA Ranking Dataset

| Column            | What it is                                       | How to read the value                                                                                                          | Example (Spain) |
| ----------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ | --------------- |
| `id_team`         | FIFA's internal team ID                          | Just an identifier — do not use as a feature                                                                                   | 43969           |
| `country_code`    | Short country code                               | Used to merge with ELO data                                                                                                    | ESP             |
| `team`            | Full country name                                | Text identifier                                                                                                                | Spain           |
| `gender`          | Men's or Women's                                 | Filter to Men for this project                                                                                                 | Men             |
| `rank`            | Current FIFA world ranking                       | Lower = better. #1 is the best team in the world                                                                               | 3               |
| `previous_rank`   | FIFA rank from the last update                   | Compare to current rank — if previous was higher number, team moved up                                                         | 2               |
| `ranking_move`    | How many places the rank changed                 | **Positive = moved up** (improving). **Negative = moved down** (declining). 0 = stayed same                                    | -1              |
| `points`          | Current FIFA ranking points                      | Higher = stronger. Use this instead of rank — it shows the gap between teams, not just position                                | 1856            |
| `previous_points` | FIFA points from last update                     | Compare to current points to compute momentum. If points went up, team is improving                                            | 1875            |
| `rated_matches`   | Matches counted in FIFA ranking calculation      | Not useful as a prediction feature — skip this                                                                                 | 58              |
| `confederation`   | Which continental federation the team belongs to | UEFA and CONMEBOL are strongest. Use as a categorical feature — teams from stronger confederations tend to be more competitive | UEFA            |
| `movement`        | Same as ranking_move but as a number             | **Positive = rising**, **negative = falling**. Same column as ranking_move, just encoded differently                           | -1              |

---

## Derived Features — Compute These Yourself

These columns do not exist in either dataset but are computed from the columns above.

| Feature                   | Formula                                | How to read the value                                                                                                   | Example (Spain vs Argentina)                                 |
| ------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `elo_diff`                | home_elo − away_elo                    | **Positive = home stronger**, negative = away stronger. Beyond ±100 is a meaningful gap                                 | 2129 − 2128 = **+1** (dead equal)                            |
| `points_diff`             | home_points − away_points              | **Positive = home stronger by FIFA**, negative = away stronger. Beyond ±50 is meaningful                                | 1856 − 1889 = **−33** (Argentina slightly stronger)          |
| `momentum`                | points − previous_points               | **Positive = team is improving**, negative = team is declining. Beyond ±15 is a real signal                             | Spain: 1856 − 1875 = **−19** (declining)                     |
| `momentum_diff`           | momentum_A − momentum_B                | **Positive = Team A trending up more**, negative = Team B trending up more                                              | −19 − 12 = **−31** (Argentina rising, Spain falling)         |
| `recent_form_diff`        | recent_form_A − recent_form_B          | **Positive = Team A in better recent form**, negative = Team B                                                          | 178 − 173 = **+5** (essentially equal)                       |
| `win_rate`                | wins / matches                         | **Higher = more dominant historically**. 0.6+ is very strong, below 0.4 is weak                                         | Spain: 462/783 = **0.59**                                    |
| `win_rate_diff`           | win_rate_A − win_rate_B                | **Positive = Team A wins more often historically**                                                                      | 0.59 − 0.34 = **+0.25** (Spain wins more often)              |
| `goals_for_per_match`     | goals_for / matches                    | **Higher = stronger attack**. Above 1.8 is excellent, below 1.2 is weak attack                                          | Spain: 1595/783 = **2.04** goals per game                    |
| `goals_against_per_match` | goals_against / matches                | **Lower = stronger defense**. Below 1.0 is excellent, above 1.5 is weak defense                                         | Spain: 699/783 = **0.89** goals conceded per game            |
| `goal_diff_per_match`     | (goals_for − goals_against) / matches  | **Positive = team wins by big margins on average**, negative = team struggles                                           | Spain: (1595−699)/783 = **+1.14** per game                   |
| `home_win_rate`           | home_wins / home_matches               | **Higher = stronger at home**. Above 0.6 is strong home form                                                            | Spain: 138/341 = **0.40**                                    |
| `elo_vs_peak`             | max_elo − elo                          | **Lower = closer to all-time best**. High value means team is far below their peak                                      | Spain: 2189 − 2129 = **60** (slightly off peak)              |
| `h2h_win_rate`            | Team A wins in H2H / total H2H matches | **Above 0.6 = strong psychological edge**, 0.4–0.6 = even, below 0.4 = opponent has the edge. Use 0.5 if no H2H history | Spain vs Argentina last 10: **0.40** (slight Argentina edge) |
