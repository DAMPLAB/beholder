'''
--------------------------------------------------------------------------------
Description:

Roadmap:
- [ ] Productization of Backend
- [x] Have Backend Startup Frontend
    - [x] Websocket Hook Up to Front End
    - [x] Dedicated datastructures to represent different aspects of the
            microscopy setup.
    - [ ] Migration to an Asynchronous Backend
    - [ ] Message Queue?
    - [ ] FastAPI?
    - [ ] Persistence of Model selection to Backend
- [ ] Initial UI Work.
    - [x] Frontend connection to backend
    - [ ] UI Elements to render the canvas.
    - [ ] UI Elements to Access the Initial Frame
    - [ ] Ability to render individual frames and push them up the websocket to
            a canvas element on the frontend.
    - [ ] Ability to select and arbitrarily design the processing pipeline.
    - [ ] Abiltiy to 'test' cellular segmentation using our currently designed
            code with an eye toward modularity in
    the future.
- [ ] Machine Learning Aspect
    - [ ] U-Net Architecture Testing
    - [ ] Dataset Curation
    - [ ] Validation, testing, hyperparameter tuning
- [ ] Better Signal Processing Aspect
    - [ ] Better ability to remove/deescalate the underlying change in diameter
            from the microfluidic device.
    - [ ] Ability to subselect efficently arbitrary geometries for different
            ROIs to prevent focusing on edges of the frame.

Written by W.R. Jackson <wrjackso@bu.edu>, DAMP Lab 2020
--------------------------------------------------------------------------------
'''
import asyncio
import json
import pathlib
import ssl
import subprocess
from typing import (
    Dict,
)

import websockets

from ds.FrameSeries import (
    load_frame_series,
    fetch_frame,
    fetch_frame_shape,
    fetch_frame_dtype,
)
from utils.logging import BLogger


REGISTERED_SUBPROCESSES = {}
PORT_NUM = 8765

LOG = BLogger()

FUNCTION_TABLE = {
    'LOAD_FRAMESERIES': load_frame_series,
    'FETCH_FRAME': fetch_frame,
    'FETCH_FRAME_SHAPE': fetch_frame_shape,
    'FETCH_FRAME_DTYPE': fetch_frame_dtype,
}
# Singletons existing on the callstack should be fine?


# ------------------------------ Startup Behavior ------------------------------
def spawn_electron_process():
    # Hardcoded for now. The script is located in the package.json for future
    # ref.
    electron_process = subprocess.Popen(
        "cd ../frontend && npm run electron:serve",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True
    )
    REGISTERED_SUBPROCESSES['electron_process'] = electron_process

# ------------------------------------------------------------------------------

# ------------------------------ Shutdown Behavior ------------------------------
def terminate_registered_subprocesses():
    for entry in REGISTERED_SUBPROCESSES:
        proc = REGISTERED_SUBPROCESSES[entry]
        proc.terminate()


# ------------------------------------------------------------------------------


# ------------------------------ Socket Parsing --------------------------------
def parse_incoming_messages(msg: dict):
    # Forgive me father for I have sinned.
    LOG.info(msg)
    cmd = msg['command']
    params = msg['params']
    LOG.info(f'{cmd=}')
    LOG.info(f'{params=}')
    func = FUNCTION_TABLE[cmd]
    # I feel like error code checking is the way I would like to do things, but
    # y'know...
    ret_message = func(**params)
    LOG.info('--------')
    return json.dumps([json.dumps({'msg': 'call', 'ret': f'{ret_message}'})])


# ------------------------------------------------------------------------------


async def main_server_routine(websocket, path):
    LOG.info('Message Recv')
    call = await websocket.recv()
    dictdump = json.loads(call)
    ret = parse_incoming_messages(dictdump)
    await websocket.send(ret)
    LOG.info('Message Sent')


if __name__ == '__main__':
    LOG.info('Welcome to Beholder!')
    LOG.info('Starting Electron Process...')
    spawn_electron_process()
    try:
        # TODO: Secure websocket SLS
        # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        # localhost_pem = pathlib.Path(__file__).with_name("localhost.pem")
        # ssl_context.load_cert_chain(localhost_pem)
        LOG.info(f'Backend running on {PORT_NUM}')
        start_server = websockets.serve(
            main_server_routine,
            "localhost",
            PORT_NUM,
            # ssl=ssl_context,
            max_size=1_000_000_000,
        )
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except KeyboardInterrupt:
        terminate_registered_subprocesses()
