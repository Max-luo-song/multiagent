"""Read-only metadata checks matching nuPlan's scenario query boundaries.

Keep the ordered_scenes/valid_scenes rule aligned with get_scenarios_from_db.
The plan records that upstream file's hash; an integration check also uses the
actual NuPlanScenarioBuilder before handing a plan to the GPU runner.
"""

from collections import defaultdict
from pathlib import Path
import sqlite3


QUERY_SOURCE = Path('third_party/nuplan-devkit/nuplan/database/nuplan_db/nuplan_scenario_queries.py')


def candidates(database, tokens=None):
    clause = ''
    args = []
    if tokens is not None:
        if not tokens:
            return []
        clause = ' AND p.token IN (' + ','.join('?' for _ in tokens) + ')'
        args = [bytes.fromhex(t) for t in tokens]
    query = '''
        WITH ordered_scenes AS (
            SELECT token, ROW_NUMBER() OVER (ORDER BY name ASC) AS row_num FROM scene
        ), num_scenes AS (
            SELECT COUNT(*) AS cnt FROM scene
        ), valid_scenes AS (
            SELECT o.token FROM ordered_scenes o CROSS JOIN num_scenes n
            WHERE o.row_num >= 3 AND o.row_num < n.cnt - 1
        )
        SELECT p.token, p.timestamp, MAX(s.type), l.logfile, l.map_version
        FROM lidar_pc p
        LEFT OUTER JOIN scenario_tag s ON p.token=s.lidar_pc_token
        INNER JOIN lidar ld ON ld.token=p.lidar_token
        INNER JOIN log l ON ld.log_token=l.token
        INNER JOIN valid_scenes vs ON p.scene_token=vs.token
        WHERE s.type IS NOT NULL
    ''' + clause + ' GROUP BY p.token,p.timestamp,l.map_version ORDER BY p.timestamp'
    with sqlite3.connect(Path(database).as_uri() + '?mode=ro&immutable=1', uri=True) as c:
        lo, hi = c.execute('SELECT min(timestamp), max(timestamp) FROM lidar_pc').fetchone()
        rows = c.execute(query, args).fetchall()
    return [dict(token=token.hex(), timestamp_us=timestamp, type=kind, log_name=log, map_name=map_name)
            for token, timestamp, kind, log, map_name in rows
            if timestamp-lo >= 2_000_000 and hi-timestamp >= 8_000_000]


def validate_plan_metadata(plan):
    groups = defaultdict(list)
    for s in plan['scenarios']:
        groups[s['database']].append(s)
    checked = 0
    for database, samples in groups.items():
        actual = {s['token']: s for s in candidates(database, [s['token'] for s in samples])}
        missing = [s['token'] for s in samples if s['token'] not in actual]
        if missing:
            raise RuntimeError('Frozen plan contains tokens excluded by the official scene-boundary '
                               f'filter in {Path(database).name}: {missing}. Prepare a corrected plan.')
        for s in samples:
            found = actual[s['token']]
            assert found['timestamp_us'] == s['timestamp_us']
            assert found['log_name'] == s['log_name']
            assert found['type'] == s['type']
            checked += 1
    return checked
