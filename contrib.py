"""Contribution calendar: realistic sample data or the real calendar from the GitHub GraphQL API."""
import datetime as dt
import json
import random
import urllib.request
from dataclasses import dataclass

LEVELS = {
    "NONE": 0,
    "FIRST_QUARTILE": 1,
    "SECOND_QUARTILE": 2,
    "THIRD_QUARTILE": 3,
    "FOURTH_QUARTILE": 4,
}


@dataclass
class Cell:
    col: int  # week, 0 = oldest
    row: int  # weekday, 0 = Sunday (top row on GitHub)
    date: dt.date
    count: int
    level: int


def _levels(counts):
    """GitHub-style levels: quartiles of the non-zero daily counts."""
    nz = sorted(c for c in counts if c > 0)
    if not nz:
        return [0] * len(counts)
    q = [nz[int(len(nz) * f)] for f in (0.25, 0.5, 0.75)]
    out = []
    for c in counts:
        if c == 0:
            out.append(0)
        elif c <= q[0]:
            out.append(1)
        elif c <= q[1]:
            out.append(2)
        elif c <= q[2]:
            out.append(3)
        else:
            out.append(4)
    return out


def sample_calendar(end=None, seed=11, density=1.0):
    """A plausible year of commits: busy weekdays, quieter weekends, sprints and a few holidays.
    `density` scales how many days have commits (for testing sparse and packed calendars)."""
    end = end or dt.date.today()
    rng = random.Random(seed)
    end_row = (end.weekday() + 1) % 7  # Python: Monday=0, GitHub: Sunday=0
    start = end - dt.timedelta(days=52 * 7 + end_row)
    n_days = (end - start).days + 1

    holidays = set()
    for _ in range(3):
        s = rng.randrange(0, n_days - 14)
        holidays.update(range(s, s + rng.randrange(4, 11)))

    counts, intensity = [], 1.0
    for i in range(n_days):
        d = start + dt.timedelta(days=i)
        intensity = min(2.4, max(0.35, intensity + rng.gauss(0, 0.13)))
        weekend = d.weekday() >= 5
        p = min(0.98, (0.30 if weekend else 0.84) * min(1.0, 0.55 + 0.3 * intensity) * density)
        c = 0
        if i not in holidays and rng.random() < p:
            mean = 3.2 * intensity * (0.55 if weekend else 1.0)
            c = 1 + int(rng.expovariate(1.0 / mean))
        counts.append(c)

    levels = _levels(counts)
    cells = [
        Cell(i // 7, i % 7, start + dt.timedelta(days=i), counts[i], levels[i])
        for i in range(n_days)
    ]
    return cells, sum(counts)


def fetch_calendar(user, token):
    """Real calendar of `user` (needs a GitHub token with read:user scope)."""
    query = """
    query($login: String!) {
      user(login: $login) {
        contributionsCollection {
          contributionCalendar {
            totalContributions
            weeks { contributionDays { date contributionCount contributionLevel weekday } }
          }
        }
      }
    }"""
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": {"login": user}}).encode(),
        headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = json.load(r)
    if "errors" in payload:
        raise RuntimeError(payload["errors"])
    cal = payload["data"]["user"]["contributionsCollection"]["contributionCalendar"]
    cells = []
    for col, week in enumerate(cal["weeks"]):
        for day in week["contributionDays"]:
            cells.append(
                Cell(
                    col,
                    day["weekday"],
                    dt.date.fromisoformat(day["date"]),
                    day["contributionCount"],
                    LEVELS[day["contributionLevel"]],
                )
            )
    return cells, cal["totalContributions"]
