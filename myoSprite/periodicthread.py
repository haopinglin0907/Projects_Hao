# periodicthread.py
# Author: Charles Lambelet
# Created: October 2019

import threading
import time
import logging


class PeriodicThread(object):
    '''Python periodic Thread using Timer with instant cancellation'''
    def __init__(self, callback=None, period=1, name=None, *args, **kwargs):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        self.callback = callback
        self.period = period
        self.stop = False
        self.current_timer = None
        self.schedule_lock = threading.Lock()
        self.time_sampled = 0

    # start the periodic thread.
    def start(self):
        self.schedule_timer()

    # run the given callback. Here the callback collects sEMG data.
    def run(self):
        if self.callback is not None:
            self.callback()

    # run the callback and then reschedule Timer (if thread is not stopped) => recursive process
    def _run(self):  # the leading underscore in the naming means that this function is private (just a convention)
        try:
            self.run()
        except Exception as e:
            logging.exception("Exception in running periodic thread")
        finally:
            with self.schedule_lock:
                if not self.stop:
                    self.schedule_timer()

    # schedule next Timer run => recursive process!
    def schedule_timer(self):
        self.current_timer = threading.Timer(self.period, self._run, *self.args, **self.kwargs)
        if self.name:
            self.current_timer.name = self.name
        self.time_sampled = time.time()
        self.current_timer.start()

    # kill the thread when not used => very important!
    def cancel(self):
        with self.schedule_lock:
            self.stop = True
            if self.current_timer is not None:
                self.current_timer.cancel()

    # block the calling thread until the thread whose join() method is called is terminated.
    def join(self):
        self.current_timer.join()