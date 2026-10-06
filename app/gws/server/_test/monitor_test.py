import gws
import gws.server.monitor as monitor


class _FailingObject:
    def periodic_task(self):
        raise ValueError('failed')


def test_failed_periodic_task_updates_last_time():
    t = monitor._Task(obj=_FailingObject(), frequency=30, lastTime=0)
    monitor.Object._run_periodic_tasks(None, [t])
    assert t.lastTime > 0
