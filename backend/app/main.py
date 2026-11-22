import asyncio
import json
import uuid
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.simulator import snapshot, routes
from app.ml import models, infer

DB="data/events.sqlite3"
faults=[]
state={"running":True,"scenario":"healthy","speed":1.0,"sim_time":0.0,"last_wall":time.monotonic()}

def db():
    os.makedirs("data",exist_ok=True)
    con=sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, sim_time REAL, kind TEXT, details TEXT)")
    con.commit()
    return con

def log(kind,details):
    with db() as con:
        con.execute("INSERT INTO events(sim_time,kind,details) VALUES(?,?,?)",(state["sim_time"],kind,json.dumps(details)))

def tick():
    now=time.monotonic()
    if state["running"]:
        state["sim_time"] += min(2,now-state["last_wall"])*state["speed"]
    state["last_wall"]=now
    return snapshot(round(state["sim_time"],2),state["scenario"],faults=faults)

@asynccontextmanager
async def lifespan(app):
    con=db();con.close()
    yield

app=FastAPI(title="LEO Satellite AI - Synthetic Research Prototype",lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173","http://127.0.0.1:5173"],allow_methods=["*"],allow_headers=["*"])

class FaultRequest(BaseModel):
    target:str=Field(min_length=1)
    kind:str
    severity:int=Field(default=80,ge=1,le=100)
    delay_s:float=Field(default=0,ge=0,le=86400)
    duration_s:float=Field(default=60,gt=0,le=86400)

class Config(BaseModel):
    running:bool=True
    speed:float=Field(1,ge=0.1,le=100)
    scenario:str="healthy"

@app.get("/api/v1/health")
def health():
    return {"status":"healthy","simulation":"synthetic","satellites":12,"ground_stations":2}

@app.get("/api/v1/snapshot")
def get_snapshot():
    return tick()

@app.post("/api/v1/simulation")
def configure(config:Config):
    if config.scenario not in ("healthy","degradation","outage","interference"):
        raise HTTPException(422,"Unknown scenario")
    tick()
    state.update(running=config.running,speed=config.speed,scenario=config.scenario)
    log("configuration",config.model_dump())
    return {"ok":True,"state":state}

@app.get("/api/v1/faults")
def list_faults():
    snap=tick()
    for fault in faults:
        fault["status"] = "scheduled" if snap["time_s"] < fault["start_s"] else "active" if snap["time_s"] < fault["end_s"] else "completed"
    return {"faults":faults,"time_s":snap["time_s"]}

@app.post("/api/v1/faults")
def create_fault(req:FaultRequest):
    if req.kind not in ("outage","overload","interference","degradation","hardware"):
        raise HTTPException(422,"Unsupported fault type")
    snap=tick()
    if req.target != "ALL" and req.target not in {l["id"] for l in snap["links"]}:
        raise HTTPException(422,"Select an active link from the current snapshot")
    if len(faults) >= 200:
        raise HTTPException(409,"Fault limit reached; clear fault history")
    fault={"id":uuid.uuid4().hex[:12],"target":req.target,"kind":req.kind,
           "severity":req.severity,"start_s":round(state["sim_time"]+req.delay_s,2),
           "end_s":round(state["sim_time"]+req.delay_s+req.duration_s,2)}
    faults.append(fault)
    log("fault_created",fault)
    return {"ok":True,"fault":fault}

@app.delete("/api/v1/faults/{fault_id}")
def stop_fault(fault_id:str):
    for fault in faults:
        if fault["id"] == fault_id:
            faults.remove(fault)
            log("fault_stopped",{"id":fault_id})
            return {"ok":True}
    raise HTTPException(404,"Fault not found")

@app.delete("/api/v1/faults")
def clear_faults():
    count=len(faults)
    faults.clear()
    log("faults_cleared",{"count":count})
    return {"ok":True,"cleared":count}

@app.post("/api/v1/reset")
def reset():
    faults.clear()
    state.update(sim_time=0.0,scenario="healthy",running=True,speed=1.0,last_wall=time.monotonic())
    log("reset",{})
    return {"ok":True}

@app.get("/api/v1/links/{link_id}/prediction")
def predict(link_id:str):
    snap=tick()
    link=next((l for l in snap["links"] if l["id"]==link_id),None)
    if not link:raise HTTPException(404,"Link not currently visible")
    return {"link_id":link_id,"result":infer(link),"note":"Synthetic baseline prediction, not validated future-event prediction"}

@app.get("/api/v1/routes")
def get_routes(source:str="SAT-01",target:str="SAT-07"):
    snap=tick()
    return {"source":source,"target":target,"routes":routes(snap,source,target)}

@app.get("/api/v1/evaluation")
def evaluation():
    return models()[4]

@app.get("/api/v1/events")
def events():
    with db() as con:
        rows=con.execute("SELECT id,sim_time,kind,details FROM events ORDER BY id DESC LIMIT 100").fetchall()
    return [{"id":r[0],"time_s":r[1],"kind":r[2],"details":json.loads(r[3])} for r in rows]

@app.websocket("/ws")
async def ws(socket:WebSocket):
    await socket.accept()
    try:
        while True:
            await socket.send_json(tick())
            await asyncio.sleep(1)
    except (WebSocketDisconnect,RuntimeError):
        pass
