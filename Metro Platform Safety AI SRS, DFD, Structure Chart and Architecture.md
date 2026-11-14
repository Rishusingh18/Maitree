# Metro Platform Safety AI: SRS, DFD, Structure Chart and Architecture

Version 0.1 draft · 14 Nov 2026 · Derived from the ideation document by @rishu (13 Nov 2026)

This document has four parts: (A) Software Requirements Specification, (B) Data Flow Diagrams, (C) Structure Chart, (D) Architecture and AI-assisted coding guide. Part D is written so it can be pasted into an AI coding assistant as project context.

---

# Part A: Software Requirements Specification (SRS)

## 1. Introduction

### 1.1 Purpose

This SRS defines the requirements for the Metro Platform Safety AI (MPSA), a camera-based advisory system that detects yellow-line crossing, track or fence intrusion, distress gestures and suspicious behaviour on metro platforms, and triggers a local buzzer and control room alerts.

### 1.2 Scope

MPSA reads existing platform CCTV (RTSP), runs detection on an edge device at the station, and sends alerts to a control room dashboard and a platform buzzer. It is **advisory**: staff stay in the loop, and MPSA never stops a train or cuts traction power by itself. The first release is a college prototype using staged video and a Raspberry Pi buzzer relay.

Out of scope: facial recognition, signalling integration, automatic train control, and certified railway safety functions.

### 1.3 Definitions

| Term | Meaning |
| --- | --- |
| PSD | Platform screen door |
| Danger strip | Region just beyond the yellow line, drawn per camera |
| Track zone | Region covering the track bed and fence line, drawn per camera |
| Track ID | Per-person identifier assigned by the tracker within one camera |
| L1 to L4 | Alert levels: Log, Warning, Critical, Silent |
| Edge node | Computer at the station running inference (Jetson or Intel box) |
| Control room | Station controller and CISF staff using the dashboard |

### 1.4 References

Metro Platform Safety AI: Ideation and Framework (13 Nov 2026); IEEE 29148 (requirements structure); India's Digital Personal Data Protection Act, 2023 (to be checked by the team for applicability).

## 2. Overall description

### 2.1 Product perspective

MPSA sits beside existing CCTV and PA systems. It consumes video and a train-state signal and produces alerts and buzzer commands. Four components: Edge Service (inference and rules), Backend (alert hub, storage, API), Dashboard (control room UI), Actuator Controller (buzzer and PA relay).

### 2.2 Product functions (summary)

1. Yellow-line guard (Module 1)
2. Fence and track jump detection (Module 2)
3. Distress gesture detection (Module 3)
4. Theft and harassment cue detection, human-review only (Module 4)
5. Alert levelling, buzzer control and acknowledgement
6. Clip recording, privacy filtering and audit logging
7. Camera and system health monitoring
8. Zone configuration and evaluation tooling

### 2.3 User classes

| User | Needs |
| --- | --- |
| Station controller | See live alerts, acknowledge, request train hold per DMRC procedure |
| CISF or platform staff | Receive L2 and L4 alerts, reach the person quickly |
| Review operator | Work through L4 low-priority queue and mark true or false |
| Administrator or engineer | Configure cameras, zones, thresholds, view health and audit logs |
| Evaluator (student team) | Run recorded video, compute metrics |

### 2.4 Operating environment

Edge: Ubuntu Linux, Python 3.11+, one GPU or accelerator, RTSP cameras. Backend: Linux server or laptop, FastAPI, MQTT broker (Mosquitto), SQLite or PostgreSQL. Dashboard: modern browser. Actuator: Raspberry Pi with relay board.

### 2.5 Constraints

- Advisory only; no control of train or traction power.
- Models run on the edge; raw video is not streamed off-site except short alert clips.
- No facial recognition.
- Open-source models and datasets for the prototype.
- Do not film real stations without permission; use a mock platform and public datasets.

### 2.6 Assumptions and dependencies

- **A1:** A train-state signal (approaching, at platform, doors closing, departed) is available. In the prototype it is simulated by a control on the dashboard or a test script publishing to MQTT.
- **A2:** Cameras are fixed; zones are drawn once per camera and re-checked on a schedule.
- **A3:** Staff are trained on the single DMRC-defined distress gesture.
- **A4:** Network between edge, broker and backend has latency low enough for the targets in section 3.4.

## 3. Specific requirements

### 3.1 Functional requirements

Priority: M = must (prototype), S = should, C = could.

#### Video ingest and perception

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-001 | The edge service shall ingest one or more RTSP streams or video files per platform. | M |
| FR-002 | The system shall detect people and 17 body keypoints in each processed frame. | M |
| FR-003 | The system shall assign a persistent Track ID per person and keep it through short occlusions. | M |
| FR-004 | The system shall reconnect to a dropped stream automatically within 10 seconds. | S |

#### Zone configuration

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-010 | An administrator shall draw the danger strip and track zone as polygons on a camera still. | M |
| FR-011 | Zones shall be stored per camera in versioned configuration. | M |
| FR-012 | The system shall support optional automatic zone proposal by segmentation, confirmed by an administrator. | C |

#### Module 1: Yellow-line guard

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-101 | The system shall flag a person whose foot keypoints lie inside the danger strip for at least N consecutive frames (default 1.5 s). | M |
| FR-102 | The system shall read train state and classify a crossing as L1 when no train is near, L2 when a train is approaching or doors are closing. | M |
| FR-103 | On L2 the system shall command a short buzzer and voice message on that platform only. | M |
| FR-104 | The system shall suppress repeat L2 alerts for the same Track ID for a configurable cool-down. | M |

#### Module 2: Fence and track jump

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-201 | The system shall raise L3 when any person is detected in the track zone for at least M frames or crosses the fence or platform edge line. | M |
| FR-202 | On L3 the system shall command a continuous buzzer and flashing light on that platform. | M |
| FR-203 | The buzzer shall continue until the track zone has been empty for a set time **and** a staff member has acknowledged the alert. | M |
| FR-204 | The L3 alert shall carry a prompt for the controller to hold the train and cut traction power per DMRC procedure; the system shall not perform these actions. | M |
| FR-205 | The system shall score pre-incident behaviour (pacing, lingering at edge, looking into the tunnel) and raise a pre-incident warning to staff. | C |

#### Module 3: Distress gesture

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-301 | The system shall detect the configured distress gesture held for about 2 seconds on a tracked person. | M |
| FR-302 | On detection the system shall raise L4 silent: no local buzzer or voice output. | M |
| FR-303 | The L4 alert shall include a clip and the camera, platform and approximate position, and shall notify the nearest staff. | M |

#### Module 4: Theft and harassment cues

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-401 | The system shall score pose sequences for snatching, pushing, close following and group surround. | S |
| FR-402 | Detections shall go to a low-priority review queue only; the system shall never trigger a buzzer or alarm from Module 4 alone. | M |
| FR-403 | A reviewer shall mark each item true or false; labels shall be stored for retraining. | S |

#### Alert engine

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-501 | The alert engine shall map each event to level L1 to L4 using the alert policy (section 3.5). | M |
| FR-502 | Each alert shall have a unique ID, timestamp, camera, platform, module, level, Track ID, confidence and clip reference. | M |
| FR-503 | The system shall deduplicate and group alerts for the same incident. | S |
| FR-504 | The system shall record alert state: new, acknowledged, resolved, false alarm. | M |

#### Clip recording and privacy

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-601 | The system shall keep a rolling buffer and save a clip from 5 s before to 10 s after each L2 to L4 event. | M |
| FR-602 | Faces shall be blurred in stored clips by default. | M |
| FR-603 | Every clip view, export and delete shall be written to an audit log with user and time. | M |
| FR-604 | Clips shall be deleted automatically after a configurable retention period. | M |
| FR-605 | The system shall not perform facial recognition or store biometric templates. | M |

#### Control room dashboard

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-701 | The dashboard shall show live tiles per platform, an alert list sorted by level and time, and a clip player. | M |
| FR-702 | Staff shall be able to acknowledge an alert, which shall be sent to the actuator controller and stored. | M |
| FR-703 | L3 alerts shall appear within the latency target with an audible dashboard tone. | M |
| FR-704 | The dashboard shall show a review queue for L4 behaviour items. | S |
| FR-705 | Users shall log in with roles: controller, reviewer, admin. | M |

#### Health and operations

| ID | Requirement | Pri |
| --- | --- | --- |
| FR-801 | The edge service shall publish a heartbeat and per-camera status (frame rate, last frame time, blind or frozen detection). | M |
| FR-802 | The system shall alert when a camera, edge node, broker or buzzer controller goes offline. | M |
| FR-803 | The system shall log false alarms per camera per day for evaluation. | S |
| FR-804 | An evaluation tool shall replay recorded video with ground-truth labels and output recall, false alarms per hour and latency. | M |

### 3.2 External interface requirements

**Video:** RTSP H.264 streams or video files, at least 15 fps, 720p or better.

**Train state:** MQTT topic or REST call carrying state and, if available, estimated time of arrival.

**Messaging:** MQTT 3.1.1 or 5 for edge-to-backend alerts and buzzer commands; WebSocket from backend to dashboard.

**Actuator:** GPIO relay on Raspberry Pi; audio output for voice message. A production station would use the existing PA instead.

**User interface:** responsive web dashboard; high-contrast critical alerts; works on a control room monitor and a tablet.

### 3.3 Safety and privacy requirements

| ID | Requirement |
| --- | --- |
| SR-01 | The buzzer must latch until empty-zone confirmation and staff acknowledgement (FR-203). |
| SR-02 | A failed or offline camera shall never be treated as "all clear"; it shall raise a health alert. |
| SR-03 | Alerts for suicide-risk cases shall direct trained staff to speak with the person; wording shall avoid method detail. |
| SR-04 | Only pose data and short blurred clips leave the edge node. |
| SR-05 | Access is role-based, every view is logged, and retention is enforced. |
| SR-06 | The system is labelled advisory; staff remain responsible for decisions. |

### 3.4 Non-functional requirements (proposed prototype targets)

| ID | Requirement | Target |
| --- | --- | --- |
| NFR-01 | Yellow-line recall | 90% or more |
| NFR-02 | False alarms per camera per hour at L2 and above | Under 1 |
| NFR-03 | Event to buzzer latency | Under 2 s |
| NFR-04 | Event to dashboard alert latency | Under 3 s |
| NFR-05 | Fence or track jump recall | 95% or more |
| NFR-06 | Gesture recall and false positive rate | 90% or more, under 2% |
| NFR-07 | Processing rate per camera on edge hardware | At least 15 fps end to end |
| NFR-08 | Uptime of edge service during a test day | 99% or more |
| NFR-09 | Extensibility: adding a new detector module | No change to tracker, alert engine or dashboard core |
| NFR-10 | Test coverage of rule and policy code | 80% or more |

These targets are proposals for a college prototype, not figures from any source.

### 3.5 Alert policy

| Level | Event | Buzzer | Control room and staff |
| --- | --- | --- | --- |
| L1 Log | Yellow line crossed, no train near | Off | Counted and clipped |
| L2 Warning | Yellow line crossed, train approaching or doors closing | Short buzzer and voice, that platform only | Alert to controller and CISF on platform |
| L3 Critical | Person on track or over fence | Continuous buzzer and flashing light | Immediate alarm; prompt to hold train and cut traction power per DMRC rules; rescue team |
| L4 Silent | Distress gesture, suspected harassment or theft | Off | Silent alert with clip and cross-camera track; nearest staff notified |

### 3.6 Data requirements

Entities: Camera, Platform, Zone, Alert, Clip, TrainState, User, AuditLog, ReviewLabel, HealthStatus. Pose data is stored as keypoint arrays without images. See Part D for schemas.

### 3.7 Verification matrix

| Requirement group | Method |
| --- | --- |
| FR-101 to FR-104 | Staged crossing videos in varied light and crowd, plus hours of normal video for false alarms |
| FR-201 to FR-205 | Mock fence with volunteer or mannequin, taped track zone |
| FR-301 to FR-303 | Staged gestures vs everyday hand movements |
| FR-401 to FR-403 | Public violence datasets; human review of false positives |
| FR-6xx, SR-04, SR-05 | Inspection of stored data, audit log checks, retention test |
| NFR-03, NFR-04 | Timestamps at camera frame, relay and dashboard |

---

# Part B: Data Flow Diagrams (DFD)

## B1. Level 0: context diagram

One process, **0 Metro Platform Safety AI**, with these external entities:

| External entity | Flows in | Flows out |
| --- | --- | --- |
| CCTV cameras | Video streams | none |
| Train state source | Train state, ETA | none |
| Administrator | Camera setup, zones, thresholds, user accounts | Health reports, audit logs |
| Station controller and staff | Acknowledgement, review labels | Alerts, clips, live video |
| Review operator | True or false labels | Review queue items |
| Buzzer and PA unit | none | Buzzer, light and voice commands |

## B2. Level 1: main processes and data stores

&#91;embedded content: Level 1 DFD · 6 processes, 4 external entities, 1 store\]

Frames become tracked people, rules combine them with zones and train state into events, and alerts drive the buzzer, clips and dashboard. Staff acknowledgement flows back into alert management.

| Process | Name | Inputs | Outputs |
| --- | --- | --- | --- |
| 1.0 | Ingest and preprocess video | Video streams | Frames |
| 2.0 | Detect, pose and track people | Frames | Tracked persons with keypoints |
| 3.0 | Evaluate safety rules (modules 1 to 4) | Tracked persons, zones, train state | Events |
| 4.0 | Manage alerts | Events, acknowledgements | Alerts, buzzer commands |
| 5.0 | Record and protect clips | Frames, alerts | Blurred clips, audit entries |
| 6.0 | Present control room dashboard | Alerts, clips, live video, health | Acknowledgements, labels |
| 7.0 | Monitor health | Heartbeats, frame status | Health alerts |
| 8.0 | Configure and evaluate | Zone and threshold input, labelled video | Config, metrics |

| Store | Name | Written by | Read by |
| --- | --- | --- | --- |
| D1 | Zone and camera configuration | 8.0 | 3.0, 6.0 |
| D2 | Alert database | 4.0 | 6.0, 8.0 |
| D3 | Clip store (blurred) | 5.0 | 6.0 |
| D4 | Audit log | 5.0, 6.0 | Administrator |
| D5 | Train state cache | Train state source | 3.0 |
| D6 | Review labels | 6.0 | 8.0 |
| D7 | Health status | 7.0 | 6.0 |

## B3. Level 2: process 3.0 (rule evaluation)

| Sub-process | Name | Reads | Produces |
| --- | --- | --- | --- |
| 3.1 | Map positions to zones | Keypoints, D1 | Zone membership per person |
| 3.2 | Yellow-line rule | Zone membership, D5 | Line-crossing event |
| 3.3 | Track and fence rule | Zone membership | Intrusion event |
| 3.4 | Pre-incident scoring | Track history | Risk score |
| 3.5 | Gesture recognition | Hand landmarks | Distress event |
| 3.6 | Behaviour recognition | Pose sequences | Review-queue event |

## B4. Data dictionary

| Data flow | Content |
| --- | --- |
| Frame | camera\_id, timestamp, image |
| Tracked person | camera\_id, timestamp, track\_id, bbox, 17 keypoints with confidence |
| Event | module, type, camera\_id, track\_id, confidence, timestamp, evidence |
| Alert | alert\_id, level, event, state, clip\_ref, station, platform |
| Buzzer command | platform\_id, mode (short, continuous, off), light, voice\_id |
| Train state | platform\_id, state, eta\_seconds, timestamp |

---

# Part C: Structure Chart

A structure chart shows the module hierarchy, who calls whom, and the data passed between modules. Numbers in brackets give the call order. Data couples are shown after each call.

&#91;embedded content: Edge Service structure chart · 1 root, 6 modules, 15 submodules\]

The numbers give call order. FaceBlur sits inside ClipRecorder and runs before any clip is saved. The text trees below list the Backend, Dashboard and Actuator.

## C1. Edge Service structure

```
EdgeMain
├── [1] ConfigLoader
│       ↓ camera list, zones, thresholds
├── [2] StreamManager (per camera)
│   ├── RTSPReader            ↑ frame
│   └── ReconnectHandler
├── [3] PerceptionPipeline
│   ├── PersonPoseDetector    ↓ frame   ↑ boxes, keypoints
│   ├── Tracker               ↓ detections   ↑ tracked persons
│   └── HandLandmarker        ↓ person crop   ↑ hand landmarks
├── [4] RuleEngine
│   ├── ZoneMapper            ↓ keypoints, zones   ↑ zone membership
│   ├── YellowLineRule        ↓ membership, train state   ↑ event
│   ├── TrackZoneRule         ↓ membership   ↑ event
│   ├── PreIncidentScorer     ↓ track history   ↑ risk score
│   ├── GestureRule           ↓ landmark sequence   ↑ event
│   └── BehaviourRule         ↓ pose sequence   ↑ event
├── [5] AlertPublisher
│   ├── AlertPolicy           ↓ event, train state   ↑ level, actions
│   ├── ClipRecorder          ↓ frame buffer, alert   ↑ clip file
│   │   └── FaceBlur          ↓ frames   ↑ blurred frames
│   └── MqttClient            ↓ alert, buzzer command
└── [6] HealthMonitor
        ↓ frame stats   ↑ heartbeat, camera status
```

## C2. Backend structure

```
BackendMain
├── MqttSubscriber       ↑ alerts, health, train state
├── AlertService
│   ├── Deduplicator
│   ├── AlertRepository   (D2)
│   └── AckHandler        ↓ ack   ↑ buzzer clear command
├── ClipService
│   ├── ClipStore         (D3)
│   └── RetentionJob
├── AuditService          (D4)
├── AuthService           roles: controller, reviewer, admin
├── ConfigService         (D1)
├── ReviewService         (D6)
├── RestApi               FastAPI routes
└── WebSocketHub          ↑ live alerts to dashboard
```

## C3. Dashboard and actuator

```
DashboardApp
├── LoginPage
├── LiveWall           platform tiles
├── AlertList          sorted by level, time
├── AlertDetail        clip player, acknowledge button
├── ReviewQueue        L4 behaviour items
├── ZoneEditor         polygon drawing on camera still
└── HealthPanel

ActuatorController (Raspberry Pi)
├── MqttListener       ↓ buzzer command
├── RelayDriver        GPIO buzzer and light
├── VoicePlayer        audio file by voice_id
└── LatchLogic         keeps continuous mode until clear command
```

## C4. Coupling rules

- Rules receive plain data objects and return events; they never touch network, disk or the camera.
- Only AlertPublisher talks to MQTT from the edge.
- Only ActuatorController touches GPIO.
- Cohesion goal: each module has one reason to change (for example, changing the yellow-line threshold touches only YellowLineRule and config).

---

# Part D: Architecture and AI-assisted coding guide

## D1. Architecture overview

Pipeline: **Camera → Edge Service (perceive, apply rules, publish) → MQTT broker → Backend (store, dedupe, serve) → Dashboard and Actuator.**

Style: edge-side modular pipeline with an event-driven message bus. Detectors are plug-in modules on one shared tracker and one alert engine, so Module 3 and 4 reuse the work of Modules 1 and 2.

&#91;embedded content: System architecture · edge node, broker, backend, dashboard, actuator\]

The edge node sends alerts to the broker and receives train state back from it for the rule engine. The broker passes alerts to the backend and buzzer commands to the actuator; staff acknowledgements travel from the dashboard back through the backend.

| Layer | Technology |
| --- | --- |
| Detection and pose | Ultralytics YOLO (v8 or 11) with Pose |
| Tracking | ByteTrack |
| Hands and gesture | MediaPipe Hands plus small GRU or LSTM (PyTorch) |
| Behaviour | YOLO-Pose keypoints into BiLSTM |
| Edge language | Python 3.11, OpenCV, NumPy, Pydantic |
| Messaging | MQTT (Mosquitto, paho-mqtt) |
| Backend | FastAPI, SQLModel or SQLAlchemy, SQLite then PostgreSQL |
| Dashboard | React with TypeScript, WebSocket |
| Actuator | Python on Raspberry Pi, gpiozero |
| Packaging | Docker Compose for backend, broker and dashboard |

## D2. Architecture decisions

| ID | Decision | Reason |
| --- | --- | --- |
| ADR-1 | Inference on the edge, send only events and short clips | Latency and privacy |
| ADR-2 | Rules are pure functions over tracked-person data | Easy to test and for AI tools to generate safely |
| ADR-3 | Alert levels decided by one AlertPolicy module from a config table | One place to change safety behaviour |
| ADR-4 | Buzzer latches until clear command from backend after staff ack | Safety rule SR-01 |
| ADR-5 | Detectors implement a common interface and register by name | NFR-09 extensibility |
| ADR-6 | Pydantic models define all messages; JSON on the wire | Contract-first, validated at both ends |
| ADR-7 | Prototype train state is simulated through the same MQTT topic a real feed would use | Swap source without code change |

## D3. Repository layout

```
mpsa/
├── README.md
├── docs/                    SRS, this document, ADRs
├── contracts/               JSON schemas and Pydantic models shared by all parts
│   └── messages.py
├── edge/
│   ├── main.py
│   ├── config/              YAML per station, zones.json per camera
│   ├── capture/             rtsp_reader.py, file_reader.py
│   ├── perception/          detector.py, tracker.py, hands.py
│   ├── rules/               base.py, yellow_line.py, track_zone.py,
│   │                        preincident.py, gesture.py, behaviour.py
│   ├── alerts/              policy.py, publisher.py, clip_recorder.py, face_blur.py
│   ├── health/              monitor.py
│   └── tests/
├── backend/
│   ├── app/                 main.py, api/, services/, models/, mqtt.py, ws.py
│   └── tests/
├── dashboard/               React app
├── actuator/                buzzer.py, latch.py, voices/
├── tools/                   zone_editor, replay_eval.py, train_state_sim.py
├── data/                    sample videos, labels (not committed if large)
└── docker-compose.yml
```

## D4. Interface contracts

Give these to the AI assistant first; every module codes against them.

### D4.1 Core data classes (Python, Pydantic)

```python
class Keypoint(BaseModel):
    x: float; y: float; conf: float

class TrackedPerson(BaseModel):
    camera_id: str
    ts: float                  # epoch seconds
    track_id: int
    bbox: tuple[float, float, float, float]
    keypoints: list[Keypoint]  # 17, COCO order

class ZoneSet(BaseModel):
    camera_id: str
    danger_strip: list[tuple[float, float]]   # polygon, image coords
    track_zone: list[tuple[float, float]]
    version: int

class TrainState(BaseModel):
    platform_id: str
    state: Literal["none", "approaching", "at_platform", "doors_closing", "departed"]
    eta_s: float | None = None
    ts: float

class Event(BaseModel):
    module: Literal["yellow_line", "track_zone", "gesture", "behaviour", "preincident"]
    type: str
    camera_id: str
    track_id: int
    confidence: float
    ts: float
    evidence: dict = {}

class Alert(BaseModel):
    alert_id: str
    level: Literal["L1", "L2", "L3", "L4"]
    event: Event
    station_id: str
    platform_id: str
    state: Literal["new", "acknowledged", "resolved", "false_alarm"] = "new"
    clip_ref: str | None = None

class BuzzerCommand(BaseModel):
    platform_id: str
    mode: Literal["off", "short", "continuous"]
    light: bool
    voice_id: str | None = None
```

### D4.2 Rule interface

```python
class Rule(Protocol):
    name: str
    def update(self, persons: list[TrackedPerson], zones: ZoneSet,
               train: TrainState | None, now: float) -> list[Event]: ...
```

Rules keep their own per-track counters internally, take `now` as an argument (no `time.time()` calls) and return events only.

### D4.3 MQTT topics

| Topic | Direction | Payload |
| --- | --- | --- |
| `mpsa/{station}/{platform}/alert` | edge → backend | Alert |
| `mpsa/{station}/{platform}/buzzer/cmd` | backend or edge → actuator | BuzzerCommand |
| `mpsa/{station}/{platform}/buzzer/state` | actuator → backend | mode, latched, ts |
| `mpsa/{station}/{platform}/train` | source → edge | TrainState |
| `mpsa/{station}/camera/{camera_id}/health` | edge → backend | fps, last\_frame\_ts, status |
| `mpsa/{station}/edge/heartbeat` | edge → backend | ts, version |

### D4.4 REST and WebSocket (backend)

| Endpoint | Purpose |
| --- | --- |
| `GET /alerts?level=&state=&platform=` | List alerts |
| `POST /alerts/{id}/ack` | Acknowledge; triggers buzzer clear if zone empty |
| `POST /alerts/{id}/resolve` | Mark resolved or false alarm |
| `GET /clips/{id}` | Stream a clip; writes audit entry |
| `GET/PUT /cameras/{id}/zones` | Read and update zones |
| `GET /health` | System health |
| `WS /ws/alerts` | Live alert push |

## D5. Key behaviours as state machines

**Yellow-line rule (per Track ID):** `Outside` → (foot in strip) → `Inside, counting` → (held ≥ N frames) → `Flagged` → emit event; → (left strip) → `Outside`. Cool-down after emit.

**Buzzer latch (per platform):** `Idle` → (L3 command) → `Sounding` → (zone empty for T seconds) → `ClearPending` → (staff ack) → `Idle`. Ack before clear stays in `Sounding`.

**Alert state:** `new` → `acknowledged` → `resolved` or `false_alarm`.

## D6. Build plan for AI-assisted coding

Build in small vertical slices, each with tests, matching the roadmap phases.

| Step | Slice | Done when |
| --- | --- | --- |
| 0 | Repo, contracts/messages.py, Docker Compose with Mosquitto | Messages validate round trip |
| 1 | file\_reader, detector, tracker | Video in, tracked persons out, boxes drawn |
| 2 | zones.json, ZoneMapper, YellowLineRule | Unit tests pass on synthetic tracks; demo video marks crossings red |
| 3 | train\_state\_sim, AlertPolicy | Table-driven tests for all L1 to L4 cases |
| 4 | Publisher, actuator, latch | Buzzer relay triggers from an alert |
| 5 | Backend alerts, ack, WebSocket | Alert appears on a test client within target latency |
| 6 | Dashboard (list, detail, ack) and ClipRecorder | End-to-end L2 demo |
| 7 | TrackZoneRule and L3 flow with latch | End-to-end L3 demo |
| 8 | HandLandmarker, GestureRule | L4 silent demo |
| 9 | FaceBlur, audit log, retention, health | Privacy checks pass |
| 10 | PreIncidentScorer, BehaviourRule, review queue | Evaluation report |

## D7. Rules for the AI coding assistant

Paste this block at the start of each session:

```
Project: Metro Platform Safety AI (advisory only, never controls trains).
Follow docs/ and contracts/messages.py exactly. Do not change message schemas
without asking. Python 3.11, type hints everywhere, Pydantic for messages.

Rules for code:
1. Rule classes are pure: no I/O, no time.time(), no global state. Take `now` as input.
2. Only AlertPublisher uses MQTT on the edge. Only actuator uses GPIO.
3. No facial recognition or face embeddings, ever. Face blur must run before any clip is saved.
4. A dead camera is never "all clear": raise a health alert.
5. Buzzer must latch until empty-zone confirmation AND staff ack.
6. Module 4 (behaviour) may only create review-queue items, never L2 or L3.
7. All thresholds come from YAML config, not constants in code.
8. Every new module comes with pytest tests, including a synthetic-track test.
9. Small diffs: one module per change. Show the file path for every file.
10. If a requirement is unclear, ask or state the assumption in a comment block.
```

### Prompt template per slice

```
Implement <module> in <path>.
Requirement IDs: <FR-xxx>. Interface: <paste D4 section>.
Inputs/outputs: <data classes>. Edge cases: <occlusion, ID switch, no train state>.
Write pytest tests covering: <normal, boundary, false-positive case>.
Do not modify other modules.
```

### Example: yellow-line rule prompt

```
Implement YellowLineRule in edge/rules/yellow_line.py (FR-101 to FR-104).
Use the Rule protocol. A person is "in strip" if either ankle keypoint with conf>0.4
is inside zones.danger_strip (use cv2.pointPolygonTest or shapely). Emit an Event
when a track has been in the strip for hold_s seconds (config, default 1.5), then
stay silent for that track for cooldown_s. Include train state in evidence.
Tests: crossing with no train, crossing with train approaching, brief 0.5 s touch
(no event), track lost and returns (counter resets), two people independent.
```

## D8. Testing and evaluation approach

- **Unit:** rules and policy with synthetic tracks and fixed clocks.
- **Integration:** replay a video file through edge to a test broker; check alerts and buzzer commands.
- **Latency:** stamp the frame time in the event and compare at actuator and dashboard.
- **Evaluation tool:** `tools/replay_eval.py` reads video plus label CSV and prints recall, false alarms per camera per hour, and latency percentiles against the NFR targets.
- **Privacy tests:** stored clips contain no clear faces; audit log has an entry for every view; retention job removes expired clips.

## D9. Risks and open points

| Item | Mitigation or decision needed |
| --- | --- |
| Real train-state feed not available | Simulate (A1); define the interface early |
| False alarms in crowds | Train-gating, hold time, per-camera tuning, metric tracking |
| Gesture adoption | Agree on one DMRC gesture; keep physical emergency buttons |
| Behaviour model false positives | Human review only (FR-402) |
| Legal and approval | Treat as advisory; check DPDP Act, 2023 and railway approvals before any real deployment |
| Suicide-related content | Staff-first response (SR-03); avoid method detail in reports |
