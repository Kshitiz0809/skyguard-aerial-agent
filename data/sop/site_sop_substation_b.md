# Site SOP — Substation-B Aerial Security & Inspection (DUMMY DOCUMENT)

> Fictional standard operating procedure written for the SkyGuard demo.
> It is not a real utility's SOP and must not be used for real flight operations.

Document ID: SOP-SUB-B-v1.3 · Owner: Site Security & Aerial Ops

## 1. Scope

1.1 This SOP covers autonomous and remotely supervised drone patrols over Substation-B, its 220 kV / 66 kV switchyard, the transformer yard, the perimeter fence and the access road.

1.2 The autonomous agent may perform observation, logging and camera actions on its own. Any change to the flight path, altitude or mission state needs approval from the Remote Pilot in Command (RPIC) or the duty security officer.

## 2. Roles

2.1 RPIC: the certified remote pilot. Has final authority over every flight action.

2.2 Duty Security Officer (DSO): receives intrusion alerts and decides whether to dispatch ground staff.

2.3 Autonomous Agent: detects anomalies, consults this SOP, proposes actions and runs only low-risk actions on its own.

## 3. Action Risk Classification

3.1 LOW-RISK actions the agent may run without approval: log_incident, zoom_camera, capture_snapshot, alert_security_team. These do not change the aircraft's trajectory.

3.2 HIGH-RISK actions that need human approval: reroute_drone, change_altitude, trigger_emergency_rtl. These change the aircraft's trajectory or mission state.

3.3 If the human operator does not respond within 60 seconds to a battery-critical RTL request, the autopilot failsafe will start RTL on its own. The agent must log this.

## 4. Perimeter Intrusion Response

4.1 Unauthorized vehicle near or inside the fence: log_incident with severity HIGH, zoom_camera to at least 4x to capture the registration plate, capture_snapshot, then alert_security_team.

4.2 Unauthorized person inside the fence: log_incident with severity CRITICAL, alert_security_team at once, and keep visual contact by loitering. Do not descend to track a person.

4.3 To get a closer look, the agent may propose reroute_drone to a standoff point at least 30 m horizontally from the intruder and no lower than 25 m AGL. The RPIC must approve the reroute.

4.4 Never descend below 15 m AGL anywhere within 50 m of energised high-voltage conductors, busbars or transformer bushings.

4.5 Do not follow an intruder outside the site geofence. Hand over to ground security.

## 5. Equipment Inspection Findings

5.1 Vegetation within 3 m of a conductor, a sagging conductor, a damaged insulator or an oil leak on a transformer: log_incident with severity MEDIUM (HIGH for an oil leak), zoom_camera and capture_snapshot.

5.2 Thermal hotspots or broken solar panels: log_incident with severity LOW or MEDIUM and capture_snapshot. No flight path change is needed.

5.3 Smoke or fire: log_incident CRITICAL, alert_security_team, and propose reroute_drone upwind to a safe standoff of at least 100 m. The RPIC must approve.

## 6. Battery and Weather Limits

6.1 Battery below 30%: do not start new detours. Finish the current task and plan to return.

6.2 Battery below 20%: trigger_emergency_rtl is MANDATORY. Cancel all other trajectory changes.

6.3 Sustained wind or gusts above 25 knots: stop the patrol and RTL. Between 18 and 25 knots, do not fly below 30 m AGL near HV lines and do not hover downwind of structures.

6.4 Visibility below 1 km or rain: RTL and log.

## 7. Altitude and Airspace

7.1 Standard patrol altitude is 40–60 m AGL. Absolute ceiling is 120 m AGL.

7.2 Stay inside the site geofence (1.5 km radius from the launch point for Substation-B).

## 8. Logging and Audit

8.1 Every detection, proposed action, approval, rejection and executed MAVLink command must be written to the mission audit log with a timestamp and the SOP clause that justifies it.

8.2 Operator overrides of a mandatory action (for example rejecting a battery-critical RTL) must be logged as SAFETY OVERRIDES and reviewed within 24 hours.
