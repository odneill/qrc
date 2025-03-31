"""
Tools for running QRC experiments
"""

import ctypes
import signal
import sys
import time
import traceback
from enum import IntEnum, auto
from itertools import accumulate
from multiprocessing import Process, Queue
from queue import Empty
from typing import Callable, Iterable, List

import numpy as np
from loguru import logger
from tqdm.auto import tqdm

_ctrlc_callbacks = {}


def _signal_handler(*args):
  """ctrl +C!"""
  for k in list(_ctrlc_callbacks.keys()):
    cb = _ctrlc_callbacks.pop(k)
    cb()
  sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def trim_memory():
  ctypes.CDLL("libc.so.6").malloc_trim(0)


# ------------------------------ Multiprocessing ----------------------------- #


class WorkerStatus(IntEnum):
  PENDING = auto()
  RUNNING = auto()
  DEAD = auto()


class Communicator:
  job_queue: Queue = None
  return_queue: Queue = None
  shutdown_queue: Queue = None
  _status: WorkerStatus = WorkerStatus.PENDING

  def __init__(self):
    self.job_queue = Queue()
    self.return_queue = Queue()
    self.shutdown_queue = Queue()


class Worker:
  com: Communicator = None
  id: int = None
  process: Process = None

  def __init__(self, id: int, com: Communicator, name: str = "worker", sentinel=None):
    self.com = com
    self.sentinel = sentinel

    self.id = id

    self.process = Process(
      target=multi_parfor_worker_fn,
      args=(self.com,),
      daemon=True,
      name=name + f"_{id}",
    )

  @property
  def status(self) -> WorkerStatus:
    if WorkerStatus.RUNNING and self.process.exitcode is not None:
      self._status = WorkerStatus.DEAD

    return self._status

  def start(self):
    self.process.start()
    self._status = WorkerStatus.RUNNING

  def kill(self):
    self.process.kill()
    self._status = WorkerStatus.DEAD

  def join(self, timeout=None):
    self.process.join(timeout=timeout)


def multi_parfor_worker_fn(com: Communicator):
  while com.shutdown_queue.empty():
    try:
      job = com.job_queue.get_nowait()
    except Empty:
      time.sleep(0.1)
    else:
      index, args, kwargs, function, logstring = job
      logger.info(logstring + f"subjob {index} starting")
      t1 = time.perf_counter()
      try:
        out = function(*args, **kwargs)
      except Exception as e:
        logger.error(
          logstring + f"subjob {index} failed: {e}\n{traceback.format_exc()}"
        )
        raise e
      logger.info(
        logstring + f"subjob {index} complete, duration: {time.perf_counter() - t1}"
      )
      com.return_queue.put((index, out))
      del (
        out,
        function,
        job,
        args,
        kwargs,
      )  # could be large, no sense keeping in mem during loop


def multi_parfor(
  iterable: Iterable,
  function: Callable | None = None,
  *,
  workers: int | List[Worker] = 8,
  ret_workers: bool = False,
  pbar: tqdm | bool | None = False,
  logstring: str | bool = False,
) -> list | tuple[list, List[Worker]]:
  """
  multiprocessing parallel loop, similar to vmap.
  Due to parallelism, not possible to share state between loop iterations.
  Run in parallel dispatched over multiple process workers.

  Parameters
  ----------
  iterable : Iterable
      An iterable object containing the inputs to the function.
  function : Callable, optional
      A callable function to apply to the inputs. If None, the first element of
      each input will be used as the function.
  workers : int or list[Worker], optional
      The number of worker processes to use. Can also be a list of Worker
      objects.
  ret_workers : bool, optional
      If True, returns the worker processes and queues.
  pbar : tqdm or bool or None, optional
      A progress bar object to display the progress of the loop. If None, a
      progress bar will be created. If False, no progress bar will be used.
  logstring : str or bool, optional
      A string to log each iteration of the loop. If none, no logging is
      performed.

  Returns
  -------
  list or tuple
      A list of outputs from the function applied to the inputs.
      If ret_workers is True, returns a tuple containing the list of outputs
      and the worker processes and queues.
  """
  if workers == 0:
    """Just do normal for loop"""
    pass

  elif isinstance(workers, int):
    com = Communicator()

    sentinel = object()
    workers = [Worker(id=i, com=com, sentinel=sentinel) for i in range(workers)]
    [w.start() for w in workers]

    def _killcb():
      com.shutdown_queue.put(True)
      [w.join(4) for w in workers]
      [w.kill() for w in workers]
      [w.join(2) for w in workers]

    _ctrlc_callbacks[sentinel] = _killcb
  else:
    sentinel = workers[0].sentinel
    com = workers[0].com

  if pbar is None or pbar is True:
    pbar = tqdm(total=len(iterable), position=0, leave=True, smoothing=0)
  if pbar is not False:
    pbar.reset(len(iterable))
    pbar.refresh()

  ls = "" if not isinstance(logstring, str) else logstring

  _outs = []
  for i, inp in enumerate(iterable):
    if function is None:
      fn = inp[0]
      args = (inp[1:],)
      kwargs = {}
    else:
      fn = function
      args = (inp,)
      kwargs = {}

    if workers == 0:
      _outs.append((i, fn(*args, **kwargs)))
    else:
      com.job_queue.put((i, args, kwargs, fn, ls))

  if workers != 0:
    try:
      while len(_outs) < len(iterable):
        if any(w.status is WorkerStatus.DEAD for w in workers):
          raise RuntimeError("A process has exited unexpectedly")
        try:
          _outs.append(com.return_queue.get_nowait())
        except Empty:
          time.sleep(0.1)
        else:
          if pbar is not False:
            pbar.update(1)
            pbar.refresh()

    except Exception as e:
      logger.critical("Interrupt recieved, shutting down workers: " + str(e))
      cb = _ctrlc_callbacks.pop(sentinel)
      cb()
      raise e

  outs = np.empty(len(iterable), dtype=object)
  for i, o in _outs:
    outs[i] = o

  if ret_workers:
    return outs, workers

  if workers != 0:
    cb = _ctrlc_callbacks.pop(sentinel)
    cb()

  trim_memory()
  return outs


def batched_multi_parfor(
  jobs,
  func,
  *,
  workers=8,
  post_func=None,
  pbars=False,
  logstring=False,
  split_factor=1,
):
  """
  Run in parallel dispatched over multiple process workers.
  Jobs are run sequentially, but distributed over the batch (x) axis.

  Takes an iterable of jobs. Each job is a tuple of (batch_args, static_args)
  Run jobs sequentially, where each job is run distributed over workers by
  splitting along the batch axis.
  The batch axis is the first axis of each argument in batch_args.

  optionally run a post_function over all of the job outputs. This is run in
  parallel over the job axis, and can be used for consolidation of outputs from
  the various parallel workers.

  """

  def noop(*a):
    pass

  if logstring is True:
    logstring = "job "

  num_w = workers if workers > 0 else 1

  _, workers_ = multi_parfor([], noop, workers=workers, ret_workers=True)

  if post_func is not None and pbars is not False:
    if pbars is None or len(pbars) < 3:
      post_pbar = tqdm(
        total=len(jobs), position=3, leave=True, smoothing=0, desc="post_func"
      )
    else:
      post_pbar = pbars[2]
  else:
    post_pbar = False

  if pbars is None:
    pbars = (
      tqdm(total=len(jobs), position=0, leave=True, desc="job_loop"),
      tqdm(total=0, position=1, leave=True, smoothing=0, desc="batch_loop"),
      post_pbar,
    )
  if pbars:
    pbars[0].reset(len(jobs))
    pbars[0].refresh()
    if len(pbars) < 3:
      pbars = (pbars[0], pbars[1], post_pbar)
  else:
    pbars = (False, False, False)

  outputs = np.empty(len(jobs), dtype=object)
  for i, (batch_args, static_args) in enumerate(jobs):
    if logstring is not False:
      t1 = time.perf_counter()
      logger.info(logstring + f"{i}/{len(jobs)} starting")
    batch_dim = len(batch_args[0])

    M, d = divmod(batch_dim, split_factor * num_w)
    if M == 0:
      bins = list(range(d + 1))
    else:
      bins = list(accumulate([M + 1] * d + [M] * (split_factor * num_w - d), initial=0))
    batch_slices = [slice(b0, b1) for b0, b1 in zip(bins[:-1], bins[1:], strict=True)]
    batched_args = [[v[s] for v in batch_args] for s in batch_slices]

    args = [(b, static_args) for b in batched_args]

    if logstring is not False:
      innerlogstring = logstring + f"{i}/{len(jobs)} "
    else:
      innerlogstring = logstring

    outputs[i], workers_ = multi_parfor(
      args,
      func,
      workers=workers_,
      ret_workers=True,
      pbar=pbars[1],
      logstring=innerlogstring,
    )
    if pbars[0]:
      pbars[0].update(1)
      pbars[0].refresh()
    if logstring is not False:
      logger.info(
        logstring + f"{i}/{len(jobs)} returned, duration: {time.perf_counter() - t1}"
      )

  if post_func is not None:
    if logstring is not False:
      innerlogstring = logstring + "post_func "
    else:
      innerlogstring = logstring

    outputs = multi_parfor(
      outputs,
      function=post_func,
      workers=workers_,
      pbar=pbars[2],
      logstring=innerlogstring,
    )  # post process and shutdown workers
  else:
    multi_parfor([], noop, workers=workers_, ret_workers=False)  # shutdown workers

  return outputs
