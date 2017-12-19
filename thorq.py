""" MIT License

Copyright (c) 2017 chongkong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import sys
import re
import subprocess
import time
import threading

THORQ_ADD = re.compile("""
Enqueue a new job:
(?:  Name: (?P<name>[^\\n]+)\n)?\
  Mode: (?P<mode>[^\\n]+)
(?:  Number of nodes: (?P<nodes>[^\\n]+)\n)?\
(?:  Number of slots: (?P<slots>[^\\n]+)\n)?\
  Device: (?P<device>[^\\n]+)
  Base directory: (?P<base_dir>[^\\n]+)
  Timeout: (?P<timeout>[^\\n]+)
  Task: (?P<task>[^\\n]+)
  Path: (?P<path>[^\\n]+)
  Command string: (?P<cmd>[^\\n]+)

Job (?P<job_id>\\d+) is enqueued.*
""", re.MULTILINE)

THORQ_STAT_ALL = re.compile("""\
(Job \\d+ \\(\\w+\\): [a-zA-Z]+\\n)+\
\\(end of the list\\).*
""", re.MULTILINE)

THORQ_STAT = re.compile(
    "Job (?P<job_id>\\d+) "
    "\\((?P<name>\\w+)\\): "
    "(?P<status>[a-zA-Z]+)\\n", re.MULTILINE)

def enqueue():
    proc = subprocess.Popen(["thorq", "--add", *sys.argv[1:]], 
                            stdout=subprocess.PIPE)
    out, _ = proc.communicate()
    m = THORQ_ADD.match(out.decode())
    job_id = m.group("job_id")
    name = m.group("name") or "task_{}".format(job_id)
    return (job_id, 
            os.path.join(m.group("path"), name + ".stdout"),
            os.path.join(m.group("path"), name + ".stderr"))

def stat(job_id):
    proc = subprocess.Popen(["thorq", "--stat-all"], stdout=subprocess.PIPE)
    out, _ = proc.communicate()
    m = THORQ_STAT_ALL.match(out.decode())
    if m is None:
        return None
    for line in m.groups():
        m = THORQ_STAT.match(line)
        if m.group("job_id") == job_id:
            return m.group("status")

def kill(job_id):
    proc = subprocess.Popen(["thorq", "--kill", job_id])
    proc.communicate()
    return

def cat(filename):
    while not os.path.isfile(filename):
        yield []
    with open(filename, "r") as f:
        while True:
            yield f.readlines() if f.readable() else []

def log_and_flush(x, end="\n"):
    sys.stderr.write(x + end)
    sys.stderr.flush()

reset='\033[0m'
darkgrey='\033[90m'
lightgrey='\033[37m'
orange='\033[33m'

def print_stdout(content):
    for line in content.splitlines():
        if len(line) > 0:
            print(f"  {lightgrey}{line}{reset}")

def print_stderr(content):
    for line in content.splitlines():
        if len(line) > 0:
            print(f"  {orange}{line}{reset}")

def monitor_output(job_id, filename, printer):
    out = cat(filename)
    status = "Enqueued"
    while status != "Running":
        status = stat(job_id)
        time.sleep(0.05)
    while status == "Running":
        status = stat(job_id)
        for _ in range(5):
            printer("".join(next(out)))
            sys.stdout.flush()
            time.sleep(0.01)

def monitor_execution(job_id, stdout, stderr):
    if os.path.isfile(stdout):
        os.remove(stdout)
    if os.path.isfile(stderr):
        os.remove(stderr)

    try:
        log_and_flush(f"{darkgrey}Enqueued task_{job_id}..", end="")
        status, cnt = "Enqueued", 0
        while status == "Enqueued":
            status = stat(job_id)
            cnt += 1
            if cnt % 20 == 0:
                log_and_flush(".", end="")
            time.sleep(0.05)
        log_and_flush(f"Launched\n---{reset}")
        t1 = threading.Thread(target=monitor_output, args=[job_id, stdout, print_stdout])
        t2 = threading.Thread(target=monitor_output, args=[job_id, stderr, print_stderr])
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        log_and_flush(f"{darkgrey}---{reset}")
    except KeyboardInterrupt:
        log_and_flush(f"\n{darkgrey}---\nAborting..", end="")
        kill(job_id)
        log_and_flush(f"{reset}")


if __name__ == '__main__':
    monitor_execution(*enqueue())
