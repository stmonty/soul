#!/usr/bin/env python3
"""Two real containers, two node daemons, one host: coordinated TCP rendezvous.
No CUDA or physical inter-node/RDMA validation. Requires cached python:3.12-slim.
"""
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
BIN = Path(os.environ.get("SOUL_BIN_DIR", ROOT / "target/debug")).resolve()
PROGRAM = r'''
import os, socket, time
rank = int(os.environ['SOUL_NODE_RANK'])
endpoint = (os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT']))
identity = os.environ['SOUL_RENDEZVOUS_ID'].encode()
if rank == 0:
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.settimeout(20)
        server.bind(endpoint)
        server.listen(1)
        conn, _ = server.accept()
        with conn:
            conn.settimeout(10)
            assert conn.recv(256) == identity
            conn.sendall(b'gang-ready')
else:
    deadline = time.monotonic() + 20
    while True:
        try:
            client = socket.create_connection(endpoint, timeout=1)
            break
        except OSError:
            if time.monotonic() > deadline: raise
            time.sleep(.1)
    with client:
        client.settimeout(10)
        client.sendall(identity)
        assert client.recv(256) == b'gang-ready'
print('rank', rank, 'rendezvous-ok', flush=True)
'''


def main():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = dict(os.environ, SOUL_TOKEN=secrets.token_hex(24))
    processes = []
    names = []
    with tempfile.TemporaryDirectory(prefix='soul-gang-') as directory:
        directory = Path(directory)
        with open(directory / 'logs', 'w+') as logs:
            def start(args):
                p = subprocess.Popen(args, env=env, stdout=logs, stderr=logs)
                processes.append(p)
                return p

            def api(path, payload=None):
                request = urllib.request.Request(base + path,
                    data=None if payload is None else json.dumps(payload).encode(),
                    headers={'Authorization': 'Bearer ' + env['SOUL_TOKEN'], 'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=5) as response:
                    body = response.read()
                    return json.loads(body) if body else None

            def wait(predicate):
                deadline = time.monotonic() + 45
                while time.monotonic() < deadline:
                    try:
                        result = predicate()
                        if result: return result
                    except OSError:
                        pass
                    time.sleep(.2)
                raise AssertionError('timed out')

            try:
                start([str(BIN / 'soul'), 'serve', str(directory / 'controller.sqlite'), str(ROOT / 'examples/two-node.json'), f'127.0.0.1:{port}'])
                state = wait(lambda: api('/v1/state'))
                for n in (1, 2):
                    node = f'node-{n}'
                    names.append(f"soul-{state['controller_id']}-{node}-gang-1")
                    start([str(BIN / 'soul-node'), base, node, str(directory / f'{node}.sqlite'), '--cpu-test', '--host-network'])
                def submit(identifier, program):
                    members = [f"soul-{state['controller_id']}-node-{n}-{identifier}-1" for n in (1, 2)]
                    if identifier != 'gang': names.extend(members)
                    api('/v1/jobs', {'id': identifier, 'pool': 'test', 'account': 'research', 'image': 'python:3.12-slim',
                        'command': ['python', '-c', program], 'nodes': 2, 'gpus_per_node': 1, 'protected_seconds': 60, 'resumable': False})
                    return members

                def finished(identifier, expected):
                    job = api('/v1/jobs/' + identifier)
                    if job['state'] in ('failed', 'completed', 'cancelled') and job['state'] != expected:
                        raise AssertionError(json.dumps(job))
                    return job if job['state'] == expected else None

                def successful(identifier):
                    members = submit(identifier, PROGRAM)
                    job = wait(lambda: finished(identifier, 'completed'))
                    assert len(job['attempts']) == 1 and len(job['attempts'][0]['placement']) == 2
                    for rank, name in enumerate(members):
                        output = subprocess.check_output(['docker', 'logs', name], text=True)
                        assert f'rank {rank} rendezvous-ok' in output

                successful('gang')
                failure_program = "import signal, sys\nsignal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n" + PROGRAM + "\nif rank == 0: sys.exit(7)\ntime.sleep(120)\n"
                failed_members = submit('rank-failure', failure_program)
                wait(lambda: finished('rank-failure', 'failed'))
                execution = api('/v1/state')['executions']['rank-failure']
                assert all(member['phase'] == 'exited' for member in execution['members'].values())
                for name in failed_members:
                    running = subprocess.check_output(['docker', 'inspect', '--format', '{{.State.Running}}', name], text=True)
                    assert running.strip() == 'false'
                successful('reuse')
                print('PASS: two daemon/container members, TCP rendezvous, rank failure stops peers, confirmed GPU reuse')
            except BaseException:
                logs.flush(); logs.seek(0); print(logs.read())
                for name in names:
                    subprocess.run(['docker', 'logs', name], check=False)
                raise
            finally:
                for p in processes:
                    if p.poll() is None: p.terminate(); p.wait(timeout=10)
                for name in names:
                    subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__': main()
