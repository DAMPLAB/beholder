'''
--------------------------------------------------------------------------------
Description:

Roadmap:
- [ ] Productization of Backend
- [x] Have Backend Startup Frontend
    - [x] Websocket Hook Up to Front End
    - [x] Dedicated datastructures to represent different aspects of the
            microscopy setup.
    - [x] Migration to an Asynchronous Backend
    - [ ] Message Queue?
    - [x] FastAPI?
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
import io
import json
import subprocess

import cv2
import websockets
from fastapi import FastAPI, Body
import numpy as np
from starlette.responses import StreamingResponse, JSONResponse
from starlette.middleware.cors import CORSMiddleware

from backend.ds.FrameSeries import (
    load_frame_series,
    retrieve_frame,
    fetch_fov_size,
    fetch_xy_size,
)
from backend.ds.Pipeline import (
    instantiate_from_yaml,
    run_channel_pipeline,
)
from backend.utils.logging import BLogger

REGISTERED_SUBPROCESSES = {}
PORT_NUM = 8765

LOG = BLogger()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


# ------------------------------- Utility Routes -------------------------------

@app.get("/echo/{input_str}")
async def echo_route(input_str: str):
    return {'echo': f'{input_str}'}


# ------------------------------ File I/O Routes -------------------------------
@app.get("/fetch_frame/{fov_num}/{frame_num}/{channel_num}")
async def fetch_frame(fov_num: int, frame_num: int, channel_num: int):
    cv2img = await retrieve_frame(fov_num, frame_num, channel_num)
    res, im_png = cv2.imencode(".png", cv2img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


@app.route("/load_dataset/{param:path}", name="path-convertor")
async def load_dataset(fp: str):
    # Returns a cv2 image array from the document vector
    fp = fp.path_params['param']
    await load_frame_series(fp)
    LOG.info('File loading complete')
    return JSONResponse({'uwu': 'uwuw'})


@app.route("/load_settings/")
async def load_dataset(fp: str):
    # Returns a cv2 image array from the document vector
    instantiate_from_yaml()
    return JSONResponse({'uwu': 'uwuw'})


@app.route("/run_individual_pipeline/")
async def load_dataset(fp: str):
    # Returns a cv2 image array from the document vector
    await run_channel_pipeline()
    return JSONResponse({'uwu': 'uwuw'})


@app.route("/get_fov_size/")
async def get_fov_size(_):
    # Returns a cv2 image array from the document vector
    fov_size = await fetch_fov_size()
    return JSONResponse({'fov_size': fov_size})


@app.route("/get_xy_size/{fov_num}")
async def get_fov_size(fov_num: int):
    # Returns a cv2 image array from the document vector
    xy_size = await fetch_xy_size(fov_num)
    return JSONResponse({'xy_size': xy_size})


# ------------------------------ Startup Behavior ------------------------------
def spawn_electron_process():
    # Hardcoded for now. The script is located in the package.json for future
    # ref.
    LOG.info('Hello')
    electron_process = subprocess.Popen(
        "pwd",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True
    )
    electron_process = subprocess.Popen(
        "cd frontend && npm run electron:serve",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True
    )
    REGISTERED_SUBPROCESSES['electron_process'] = electron_process
    LOG.info('Goodbye')


# ------------------------------ Shutdown Behavior ------------------------------
def terminate_registered_subprocesses():
    for entry in REGISTERED_SUBPROCESSES:
        proc = REGISTERED_SUBPROCESSES[entry]
        proc.terminate()


LOG.info('Welcome to Beholder!')
LOG.info('Starting Electron Process...')
spawn_electron_process()
