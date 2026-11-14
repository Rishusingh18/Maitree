# Metro Platform Safety AI: Ideation and Framework

Nov 13, 2026 · @rishu

A camera-based system that detects yellow-line crossing, track jumps and distress signals on metro platforms, then triggers a platform buzzer and a control room alert.

## Problem and opportunity

Platform edge incidents are a recurring safety problem in Delhi Metro, and cameras already cover every platform, so the missing piece is software that watches them and reacts in seconds.

- One report counted 37 suicide-related incidents in six months at Delhi Metro stations, with 11 deaths and 14 people hospitalised after CISF intervention.
- Another report put the figure at 50 cases in one year, 18 of them on the Yellow Line, and counted 46 deaths from 2015 to 2018.
- Platform screen doors (PSDs) solve this only where they exist: DMRC has said it has no proposal to fit them across the whole network, and the Yellow Line had them at just a handful of stations.
- DMRC runs roughly 24,000 CCTV cameras and has begun upgrading to IP cameras with video analytics, so the hardware base is there.

Your idea fills the gap for stations without PSDs and adds a layer on top for stations that have them: **see the event, sound the local alarm, tell the control room, all without waiting for a human to notice the screen.**

## What already exists

Yellow-line and track-intrusion detection is already in pilots and products worldwide, so the novelty of this project is the combination (one platform system covering edge, fence, gesture and harassment) and the Indian context, not any single detector.

| System | What it does | Response | Source |
| --- | --- | --- | --- |
| Hong Kong MTR (East Rail, 13 stations, about 400 platform cameras) | Detects intrusion, person or object on track, trespassing, beyond-yellow-line, collapse, wandering, animals | Critical alarm: staff trip the train with the emergency stop. Alert level: automatic public address for beyond-yellow-line | [MTR talk, 2024](https://international-railway-safety-council.com/wp-content/uploads/2025/04/Billy-Wai-Pan-Tam_Autonomous-Detection-and-Alert-with-Cutting-edge-Computer-Vision-Enhancing-Platform-Safety.pdf) |
| New York MTA | Two-station pilot of AI track intrusion detection, including pre-intrusion behaviour such as leaning over the edge | Alerts to train operators, station staff and control centre | [Report](https://okdiario.com/techy/en/new-yorks-subway-is-now-using-ai-to-catch-riders-in-restricted-zones-even-those-who-paid-their-fare/6520/) |
| RailSight YellowLine (UK) | Detects the yellow line, whether a train is in transit, and who is over the line | Alerts drivers and staff if someone is stuck in the gap | [DIT listing](https://www.great.gov.uk/international/buy-from-the-uk/find-a-supplier/supplier/case-study/2853) |
| LTM TransitWatch (India) | YOLO 11 people detection on Intel edge devices: yellow-line crossing, unattended bags, wrong-direction movement, intruders | Red box, count per camera and platform, clip saved with logs | [LTM deck](https://www.ltm.com/content/dam/ltimcorporatewebsite/refresh-images/pdf/transit-watch-ltm-2026.pdf.html) |
| DMRC CCTV upgrade (76 locations) | Video analytics to flag a person on the track, overcrowding, loitering | Alerts and events shown on control room video walls | [PTI via ThePrint](https://theprint.in/india/dmrc-to-upgrade-cctv-network-at-76-locations-with-ai-based-surveillance/3023608/) |
| Delhi Safe City | AI video analytics for crimes against women, abandoned objects and crowds | Real-time alerts to a command centre | [PTI via ThePrint](https://theprint.in/india/3500-ai-enabled-cameras-to-guard-delhi-under-safe-city-project-from-oct-1/2688353) |

Two research strands matter most for your extensions. A Canadian study scores suicide risk per person from pose and movement, using three platform zones (wall, yellow line, tunnel end) to catch pacing and lingering ([Unite.AI summary](https://www.unite.ai/anticipating-and-preventing-metro-platform-tragedies-with-ai/)). And the "Signal for Help" hand gesture has been turned into real-time detectors with MediaPipe and small CNNs ([PMC paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9826541)).

## Proposed system: four modules on one pipeline

One camera feed, one person tracker and one alert engine serve four detectors, so each extra module costs little once the first works.

| Module | Trigger | How it detects | Local response | Control room |
| --- | --- | --- | --- | --- |
| 1. Yellow-line guard | Person's feet beyond the line **while a train is approaching or the doors are closed to boarding** | Fixed line mask per camera, YOLO person tracking, foot keypoint inside the danger strip for a set number of frames | Buzzer and voice message on that platform only | Alert with camera and platform ID plus a short clip |
| 2. Fence and track jump | Person crosses the fence or platform edge, or any person detected in the track zone | Track-zone mask, tracking, sudden vertical or edge-crossing motion | Continuous buzzer and flashing light until the zone is clear | Critical alarm: "someone jumped", request to hold or stop the train and cut traction power as per DMRC procedure |
| 3. Distress signal | Person shows the Signal for Help or a DMRC-defined gesture, held for about two seconds | MediaPipe hand landmarks plus a small sequence classifier on a tracked person crop | None, to avoid alerting an aggressor | Silent alert with a clip, nearest CISF staff notified |
| 4. Theft and harassment cues | Snatching motion, pushing, sustained close following, a person surrounded by a group | Pose keypoints (YOLO-Pose) into an LSTM or transformer; a person-in-group-circle rule | Optional PA message | Low-priority review queue, human decides |

Build order matters: modules 1 and 2 are geometry plus tracking and can reach good accuracy in a college prototype. Module 3 is a well-studied gesture problem. Module 4 is the least reliable and should only ever ask a human to look, never act alone.

## How it fits together

&#91;embedded content: system architecture · 1 camera, 4 detectors, 3 responses\]

The yellow-line buzzer and fence alarm go to the platform and the control room, the distress gesture goes only to staff, and behaviour cues wait for a human to review.

## Alert and response workflow

Alerts come in four levels so that the loud response (buzzer) is used only where it helps, and quiet cases stay quiet. The Hong Kong MTR uses the same idea: a critical alarm leads to an emergency train stop, a lower alert to a public address message.

| Level | Event | Platform buzzer | Control room and staff |
| --- | --- | --- | --- |
| L1 Log | Yellow line crossed, no train near | Off | Counted and clipped for later review |
| L2 Warning | Yellow line crossed with a train approaching | Short buzzer and voice message on that platform only | Alert to station controller and CISF on the platform |
| L3 Critical | Person on track or over the fence, "someone jumped" | Continuous buzzer and flashing light | Immediate alarm; request to stop or hold the train and cut traction power per DMRC rules; rescue team sent |
| L4 Silent | Distress gesture, suspected harassment or theft | Off, so the aggressor is not alerted | Silent alert with a clip and the person's track across cameras; nearest CISF or Kali Squad staff notified |

**Buzzer rule for your idea:** the buzzer keeps sounding until the camera sees the track zone empty for a few seconds **and** a staff member acknowledges it. Camera-only reset is risky, because a person lying under the platform edge can be missed.

## Tech stack, models and datasets

A working prototype needs only open-source models and one GPU or edge board; the hard part is data, not code.

| Layer | Suggested choice | Why |
| --- | --- | --- |
| Person detection and pose | YOLOv8 or YOLO11 with the Pose variant (17 keypoints) | One model gives boxes and skeletons; used by the platform and violence papers found above |
| Tracking | ByteTrack or DeepSORT | Keeps an ID per person so rules like "over the line for 1.5 s" work |
| Platform zones | YOLOv8-seg or SegNet mask of platform, yellow strip and track; or a line drawn once in the UI | A segmentation approach is used in an open metro project; LTM draws the line once per camera |
| Gesture | MediaPipe Hands landmarks into a small LSTM or GRU | Reported 96.43% accuracy for the help-signal sequence with MediaPipe, and 91.25% for a dual-CNN design |
| Behaviour (module 4) | YOLO-Pose keypoints into BiLSTM or transformer | Pose input is lighter and more private than raw video |
| Edge compute | NVIDIA Jetson or Intel Core Ultra box per station | LTM runs YOLO 11 on Intel edge hardware |
| Messaging and backend | RTSP in, MQTT or WebSocket for alerts, FastAPI | Low latency and easy to demo |
| Control room | React dashboard with live tile, alert list, clip player, acknowledge button | Same shape as the TransitWatch dashboard |
| Buzzer and PA | Relay on a Raspberry Pi for the prototype; real stations would plug into the existing PA | Cheap to demo, realistic path to deploy |

**Data sources**

- Metro platform frames: the Roboflow "metro-detection" set behind an open [subway line-crossing project](https://github.com/smirnovlad/subwaydetectionassistant) (1,125 annotated frames after augmentation).
- Violence and fights: RWF-2000 (2,000 surveillance clips) and Hockey Fight (1,000 clips), as used in an [open YOLO-pose plus LSTM project](https://awesome.ecosyste.ms/projects/github.com%2Fvaliantlynx%2Fgodseye); a [synthetic CCTV aggressive-pose dataset](https://huggingface.co/datasets/Simuletic/CCTV_Aggressive_Poses_Fight_Detection_Dataset/blob/main/README.md) under CC-BY-4.0.
- Help gesture: a 200-video "Signal for Help" dataset from the [Intelligent Systems with Applications paper](https://doaj.org/article/0f32da6745664c219859498d483ad427), plus the 27-class Jesture set used for pre-training.
- Your own data: record staged clips in a mock platform (tape line, volunteers, a chair as fence) at the angles of real CCTV. Do not film real metro stations without permission.

## Risks, privacy, and what to change in the original idea

Five changes make the idea stronger and easier to defend in a viva or a pitch.

1. **Gate the yellow-line buzzer by train position.** On a crowded platform people stand near the line while waiting, so a buzzer that fires on every crossing will be ignored within days. Fire it only when a train is arriving or leaving.
2. **Pair the buzzer with a train response.** A buzzer tells people on the platform; it does not stop a train. For fence or track cases, the alert must reach the controller fast enough to hold the train and cut traction power, as the MTR critical-alarm process does.
3. **Add pre-incident warning.** The Canadian study scores each person from pacing, lingering at the edge and looking into the tunnel, minutes before a jump. This turns "someone jumped" into "someone may jump", which is the real prevention.
4. **Do not call it sign language.** The Signal for Help is a single internationally promoted hand gesture, not a sign language, and it works only if responders know it. Define one clear DMRC gesture, teach CISF staff, and keep the physical emergency buttons as backup.
5. **Treat harassment and theft as human-review alerts.** Models trained on fight clips confuse hugs, high-fives and sports with violence; the literature reports these false positives. Send such cases to an operator queue, never straight to an alarm.

**Risks to plan for**

- **False alarms:** rush-hour crowds, people hidden behind others, glare and tilted camera angles. Track false alarms per camera per day, not only accuracy.
- **Privacy and misuse:** in 2013 over 250 clips from Delhi Metro station cameras were edited and posted on adult sites, which is why access control matters. Run models on the edge, keep only pose data and short clips, blur faces, avoid facial recognition, log every clip view, and set a retention limit. Check the requirements of India's Digital Personal Data Protection Act, 2023.
- **Camera uptime:** reports have flagged defunct Safe City cameras, so the system needs health checks that alert when a camera goes blind.
- **Safety approvals:** a real railway deployment needs signalling and safety certification. Position the prototype as advisory, with staff always in the loop.
- **Sensitive topic:** jump prevention involves suicide. Alerts should send trained staff to talk to the person, not just to punish crossing, and any public write-up should avoid method detail.

## Evaluation and roadmap

The prototype is judged on how many real events it catches and how rarely it cries wolf. The targets below are proposals for a college prototype, not figures from any source.

| Metric | Proposed target | How to measure |
| --- | --- | --- |
| Yellow-line recall | 90% or more | Staged crossings across lighting and crowd levels |
| False alarms per camera per hour | Under 1 at L2 and above | Run on hours of recorded normal platform video |
| End-to-end latency, event to buzzer | Under 2 seconds | Timestamp at camera and at relay |
| Control room alert latency | Under 3 seconds | Timestamp at dashboard |
| Gesture recall and false positive rate | 90% or more, under 2% | Staged gestures vs everyday hand movements |
| Fence or track jump recall | 95% or more | Mock-up fence and mannequin or volunteer crossing a taped zone |

| Phase | Goal | Deliverable |
| --- | --- | --- |
| 1. Foundations (weeks 1 to 3) | Person detection, tracking, and a drawn yellow line | Video demo that marks people over the line in red |
| 2. Alerts (weeks 4 to 5) | Train-gated rule, buzzer relay, dashboard | Buzzer sounds on one "platform"; alert with clip appears on the dashboard |
| 3. Fence and jump (weeks 6 to 7) | Track-zone intrusion and continuous buzzer with staff acknowledge | L3 critical flow end to end |
| 4. Distress gesture (weeks 8 to 9) | Help-signal detector and silent alert path | L4 silent alert demo |
| 5. Behaviour and privacy (weeks 10 to 12) | Pre-incident scoring, harassment review queue, face blur, access log | Evaluation report with the metrics above |

Plans of this size usually fit a semester; phases 1 to 3 alone already make a strong project.

## Sources

Figures come from search results read on 13 Nov 2026; several news items are older and undated in the snippets, so verify numbers before putting them in a report.

- [Unite.AI: Anticipating and preventing metro platform tragedies with AI](https://www.unite.ai/anticipating-and-preventing-metro-platform-tragedies-with-ai/)
- [MTR Corporation: Autonomous detection and alert for platform safety (2024)](https://international-railway-safety-council.com/wp-content/uploads/2025/04/Billy-Wai-Pan-Tam_Autonomous-Detection-and-Alert-with-Cutting-edge-Computer-Vision-Enhancing-Platform-Safety.pdf)
- [New York MTA track intrusion pilot](https://okdiario.com/techy/en/new-yorks-subway-is-now-using-ai-to-catch-riders-in-restricted-zones-even-those-who-paid-their-fare/6520/)
- [LTM TransitWatch](https://www.ltm.com/content/dam/ltimcorporatewebsite/refresh-images/pdf/transit-watch-ltm-2026.pdf.html)
- [RailSight YellowLine](https://www.great.gov.uk/international/buy-from-the-uk/find-a-supplier/supplier/case-study/2853)
- [Open subway line-crossing project (YOLOv8 pose, SegNet)](https://github.com/smirnovlad/subwaydetectionassistant)
- [Sunday Guardian: Delhi Metro suicide bids](https://sundayguardianlive.com/news/delhi-metro-suicide-bids-see-11-deaths-6-months/amp)
- [Delhi Metro PSD and 50 cases in a year](https://metrorailnews.in/?p=8972)
- [Lok Sabha reply: no PSD proposal for entire DMRC network](https://eparlib.nic.in/bitstream/123456789/679205/1/3217.pdf)
- [PTI: DMRC CCTV upgrade at 76 locations](https://theprint.in/india/dmrc-to-upgrade-cctv-network-at-76-locations-with-ai-based-surveillance/3023608/)
- [DMRC on CCTV network (about 24,000 cameras)](https://www.devdiscourse.com/article/business/3683313-delhi-metros-robust-cctv-network-ensuring-commuter-safety)
- [Delhi Safe City AI cameras](https://theprint.in/india/3500-ai-enabled-cameras-to-guard-delhi-under-safe-city-project-from-oct-1/2688353)
- [Trading privacy for safety: women and surveillance in Delhi](https://www.missingperspectives.com/posts/being-a-woman-in-most-surveilled-city-in-the-world)
- [Signal for Help dataset and CNN workflow (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9826541)
- [Signal for Help with MediaPipe, 96.43% accuracy](https://ubmr.hds.hebis.de/EdsRecord/buh%252C188454834)
- [Pose-based violence detection, Politecnico delle Marche](https://iris.univpm.it/retrieve/174a7815-34f0-4fc0-ac23-1c031c133ad1/Spotting%20the%20Aggressor%20Pose-Based%20Violence%20Detection%20Through%20.pdf)
- [Pose estimation violence detection, Cranfield](https://dspace.lib.cranfield.ac.uk/handle/1826/21077)
