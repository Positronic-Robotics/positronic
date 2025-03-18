# MIT License

# Copyright (c) 2024 Jimmy Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import atexit
import os
from pathlib import Path

def create_pid_file(name):
    # Check if PID file already exists
    pid_file_path = Path(f'/tmp/{name}.pid')
    if pid_file_path.exists():
        # Get PID of other process from lock file
        with open(pid_file_path, 'r', encoding='utf-8') as f:
            pid = int(f.read().strip())

        # Check if PID matches current process
        if pid != os.getpid():
            # PID does not match current process, check if other process is still running
            try:
                os.kill(pid, 0)
            except OSError:
                print(f'Removing stale PID file (PID {pid})')
                pid_file_path.unlink()
            else:
                raise Exception(f'Another instance of the {name} is already running (PID {pid})')

    # Write PID of current process to the file
    pid_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pid_file_path, 'w', encoding='utf-8') as f:
        f.write(f'{os.getpid()}\n')

    # Register cleanup function to remove PID file upon exit
    atexit.register(remove_pid_file, pid_file_path)

def remove_pid_file(pid_file_path):
    # Remove PID file if it corresponds to the current process
    if pid_file_path.exists():
        with open(pid_file_path, 'r', encoding='utf-8') as f:
            pid = int(f.read().strip())
        if pid == os.getpid():
            pid_file_path.unlink()
