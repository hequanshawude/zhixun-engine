"""Read numeric validation measurements returned by the host."""
import math


def primary_metric(task):
    return 'macro_f1' if task == 'classification' else 'rmse'


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def read_metrics(reply, task):
    if not isinstance(reply, dict):
        return None
    primary = primary_metric(task)
    for container in (reply, reply.get('summary'), reply.get('validation')):
        if not isinstance(container, dict):
            continue
        for metrics in (container.get('metrics'), container.get('validation_metrics'), container):
            if isinstance(metrics, dict) and _finite(metrics.get(primary)):
                return {key: value for key, value in metrics.items() if _finite(value)}
    return None


def accept_candidate(parent, candidate, task):
    current = read_metrics(parent, task)
    proposed = read_metrics(candidate, task)
    if proposed is None:
        return False, 'candidate platform measurement missing'
    if current is None:
        return False, 'parent platform measurement missing'
    primary = primary_metric(task)
    improved = proposed[primary] > current[primary] if task == 'classification' else proposed[primary] < current[primary]
    return improved, 'host measurement improved' if improved else 'host measurement did not meet gate'
