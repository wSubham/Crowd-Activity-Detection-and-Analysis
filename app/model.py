# model.py

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


# ============================================================
#                 MODEL 1: ACTIVITY ONLY
#   (Based on final_activity_only.py with per-frame API)
# ============================================================

class ActivityModel:
    """
    Human Activity Detection using YOLO Pose.
    Actions: Sitting, Standing, Walking, Running
    """

    def __init__(self,
                 model_name: str = "models/yolo11s-pose.pt",
                 fps: float = 30.0):
        self.model = YOLO(model_name)
        self.fps = fps if 0 < fps <= 120 else 30.0

        # thresholds (same as your script)
        self.TH_STAND = 0.35
        self.TH_RUN = 1.6

        self.track_history = defaultdict(
            lambda: deque(maxlen=int(self.fps / 2))
        )
        self.prev_positions: Dict[int, Tuple[float, float]] = {}

    @staticmethod
    def calculate_angle(a, b, c):
        """Calculates angle between three points (e.g., Hip, Knee, Ankle)"""
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return 360 - angle if angle > 180.0 else angle

    def process_frame(self, frame):
        """
        Process one frame and return:
        - annotated_frame
        - counts dict: {Sitting, Standing, Walking, Running}
        """
        h, w, _ = frame.shape
        counts = {"Sitting": 0, "Standing": 0, "Walking": 0, "Running": 0}

        results = self.model.track(
            frame, persist=True, tracker="bytetrack.yaml", verbose=False
        )

        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            keypoints = results[0].keypoints.data.cpu().numpy()

            for box, track_id, kp in zip(boxes, track_ids, keypoints):
                x1, y1, x2, y2 = box
                height = max(1, y2 - y1)

                # Center point -> ankles if available
                if kp[15, 2] > 0.6 and kp[16, 2] > 0.6:
                    cx, cy = (kp[15, 0] + kp[16, 0]) / 2, (kp[15, 1] + kp[16, 1]) / 2
                else:
                    cx, cy = (x1 + x2) / 2, y2

                # --- speed ---
                current_speed = 0.0
                if track_id in self.prev_positions:
                    px, py = self.prev_positions[track_id]
                    dist_pixels = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    current_speed = (dist_pixels / height) * self.fps

                self.prev_positions[track_id] = (cx, cy)

                self.track_history[track_id].append(current_speed)
                avg_speed = float(np.mean(self.track_history[track_id]))

                # --- knee angle ---
                avg_knee_angle = 180
                if kp[13, 2] > 0.5 and kp[14, 2] > 0.5:
                    l = self.calculate_angle(kp[11, :2], kp[13, :2], kp[15, :2])
                    r = self.calculate_angle(kp[12, :2], kp[14, :2], kp[16, :2])
                    avg_knee_angle = (l + r) / 2

                action = "Unknown"
                color = (255, 255, 255)  # default white

                if avg_knee_angle < 135:
                    action = "Sitting"
                    color = (0, 0, 255)      # Red
                elif avg_speed > self.TH_RUN:
                    action = "Running"
                    color = (0, 255, 255)    # Yellow
                elif avg_speed > self.TH_STAND:
                    action = "Walking"
                    color = (255, 255, 0)    # Cyan
                else:
                    action = "Standing"
                    color = (0, 255, 0)      # Green

                counts[action] += 1

                # drawing
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              color, 2)
                cv2.putText(frame, action,
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            color, 2)

        # --- Top dashboard bar ---
        cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

        dashboard_items = [
            ("SIT", counts["Sitting"], (0, 0, 255)),      # Red
            ("STD", counts["Standing"], (0, 255, 0)),     # Green
            ("WALK", counts["Walking"], (255, 255, 0)),   # Cyan
            ("RUN", counts["Running"], (0, 255, 255))     # Yellow
        ]

        x_offset = 30
        font = cv2.FONT_HERSHEY_DUPLEX

        for label, count, col in dashboard_items:
            text = f"{label}: {count}"
            (text_w, _), _ = cv2.getTextSize(text, font, 0.8, 2)

            cv2.putText(frame, text, (x_offset, 40),
                        font, 0.8, col, 2)
            cv2.line(frame,
                     (x_offset + text_w + 20, 15),
                     (x_offset + text_w + 20, 45),
                     (100, 100, 100), 1)
            x_offset += text_w + 40

        return frame, counts


# ============================================================
#      MODEL 2: CROWD DETECTION + HEATMAP + GROUP ALERT
#   (Streamlit-friendly wrapper around final_final_pakka.py)
# ============================================================

@dataclass
class Config:
    trajectory_length: int = 15
    distance_threshold: float = 110.0
    size_ratio_threshold: float = 0.5
    frechet_threshold: float = 80.0

    crowd_threshold: int = 4
    alert_trigger_seconds: float = 2.0

    confidence_threshold: float = 0.45

    tracker_max_age: int = 30
    tracker_n_init: int = 2
    tracker_max_iou_distance: float = 0.7

    default_fps: float = 25.0
    max_fps: float = 120.0

    box_thickness: int = 3
    text_scale: float = 0.6
    text_thickness: int = 2
    trajectory_thickness: int = 2
    alarm_flash_rate: int = 10

    heatmap_decay: float = 0.90
    heatmap_radius: int = 30
    heatmap_intensity: float = 3.0
    heatmap_alpha: float = 0.6

    color_white: Tuple[int, int, int] = (255, 255, 255)
    color_red: Tuple[int, int, int] = (0, 0, 255)
    color_yellow: Tuple[int, int, int] = (0, 255, 255)
    color_black: Tuple[int, int, int] = (0, 0, 0)
    color_orange: Tuple[int, int, int] = (0, 165, 255)


@dataclass
class Person:
    id: int
    bbox: Tuple[int, int, int, int]
    foot_point: np.ndarray
    diagonal_length: float


@dataclass
class Group:
    members: List[int]
    is_crowd: bool
    crowd_duration: float = 0.0
    alarm_triggered: bool = False
    group_id: Optional[str] = None


def calculate_foot_point(bbox):
    x1, y1, x2, y2 = bbox
    height = y2 - y1
    return np.array([(x1 + x2) / 2, y1 + 0.9 * height])


def calculate_diagonal_length(bbox):
    x1, y1, x2, y2 = bbox
    return float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))


def discrete_frechet_distance(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return float("inf")

    P = list(P)
    Q = list(Q)
    ca = np.full((len(P), len(Q)), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        dist = np.linalg.norm(P[i] - Q[j])
        if i == 0 and j == 0:
            ca[i, j] = dist
        elif i > 0 and j == 0:
            ca[i, j] = max(c(i - 1, 0), dist)
        elif i == 0 and j > 0:
            ca[i, j] = max(c(0, j - 1), dist)
        else:
            ca[i, j] = max(
                min(c(i - 1, j), c(i - 1, j - 1), c(i, j - 1)),
                dist
            )
        return ca[i, j]

    return c(len(P) - 1, len(Q) - 1)


def generate_group_color(group_idx: int, seed: int = 99):
    np.random.seed(group_idx * seed)
    return tuple(np.random.randint(100, 255, 3).tolist())


def create_group_id(member_ids: List[int]) -> str:
    return "-".join(map(str, sorted(member_ids)))


class GroupTimerTracker:
    def __init__(self, alert_threshold: float):
        self.alert_threshold = alert_threshold
        self.group_timers: Dict[str, int] = {}
        self.active_alarms: Set[str] = set()

    def update(self, groups: List[Group],
               people: List[Person],
               fps: float) -> List[Group]:
        current_group_ids = set()
        updated_groups = []

        for group in groups:
            person_ids = [people[idx].id for idx in group.members]
            group_id = create_group_id(person_ids)
            group.group_id = group_id

            if group.is_crowd:
                current_group_ids.add(group_id)
                if group_id not in self.group_timers:
                    self.group_timers[group_id] = 0
                self.group_timers[group_id] += 1

                group.crowd_duration = self.group_timers[group_id] / fps
                if group.crowd_duration >= self.alert_threshold:
                    group.alarm_triggered = True
                    self.active_alarms.add(group_id)
                else:
                    group.alarm_triggered = False

            updated_groups.append(group)

        expired = set(self.group_timers.keys()) - current_group_ids
        for gid in expired:
            del self.group_timers[gid]
            self.active_alarms.discard(gid)

        return updated_groups

    def get_active_alarm_count(self) -> int:
        return len(self.active_alarms)

    def has_any_alarm(self) -> bool:
        return len(self.active_alarms) > 0


class HeatmapGenerator:
    def __init__(self, width: int, height: int, config: Config):
        self.config = config
        self.heatmap = np.zeros((height, width), dtype=np.float32)

    def update(self, people: List[Person], groups: List[Group]):
        self.heatmap *= self.config.heatmap_decay

        crowd_indices = set()
        for g in groups:
            if g.is_crowd:
                crowd_indices.update(g.members)

        for idx, p in enumerate(people):
            fp = p.foot_point.astype(int)
            cx, cy = fp[0], fp[1]
            if 0 <= cx < self.heatmap.shape[1] and 0 <= cy < self.heatmap.shape[0]:
                intensity = self.config.heatmap_intensity
                if idx in crowd_indices:
                    intensity *= 2.0
                cv2.circle(self.heatmap,
                           (cx, cy),
                           self.config.heatmap_radius,
                           intensity,
                           -1)

    def generate_overlay(self, frame: np.ndarray) -> np.ndarray:
        heat_norm = cv2.normalize(
            self.heatmap, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        heat_color = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(
            frame, 1 - self.config.heatmap_alpha,
            heat_color, self.config.heatmap_alpha,
            0
        )
        return overlay


class TrajectoryTracker:
    def __init__(self, max_length: int = 15):
        self.trajectories: Dict[int, deque] = {}
        self.max_length = max_length

    def update(self, person: Person):
        if person.id not in self.trajectories:
            self.trajectories[person.id] = deque(maxlen=self.max_length)
        self.trajectories[person.id].append(person.foot_point)

    def get(self, pid: int):
        return list(self.trajectories.get(pid, []))

    def cleanup(self, active_ids: Set[int]):
        inactive = set(self.trajectories.keys()) - active_ids
        for pid in inactive:
            del self.trajectories[pid]


class GroupDetector:
    def __init__(self, config: Config, traj_tracker: TrajectoryTracker):
        self.config = config
        self.traj_tracker = traj_tracker

    def calculate_interaction_score(self, p1: Person, p2: Person):
        distance = np.linalg.norm(p1.foot_point - p2.foot_point)
        size_ratio = min(p1.diagonal_length, p2.diagonal_length) / \
                     max(p1.diagonal_length, p2.diagonal_length)

        traj1 = self.traj_tracker.get(p1.id)
        traj2 = self.traj_tracker.get(p2.id)
        len1 = len(traj1)
        len2 = len(traj2)

        if len1 >= 5 and len2 >= 5:
            frechet = discrete_frechet_distance(traj1, traj2)
        else:
            # Not enough history yet -> treat as 0 so condition can pass
            frechet = 0.0

        return distance, size_ratio, frechet, len1, len2

    def build_adjacency_matrix(self, people: List[Person]):
        n = len(people)
        adj = np.zeros((n, n), dtype=int)
        if n == 0:
            return adj
        for i in range(n):
            for j in range(i + 1, n):
                d, r, f, len1, len2= self.calculate_interaction_score(people[i], people[j])
                if d >= self.config.distance_threshold:
                    continue
                if r <= self.config.size_ratio_threshold:
                    continue

                # If we don't have at least 5 points for both, ignore Frechet
                if len1 < 5 or len2 < 5:
                    adj[i, j] = adj[j, i] = 1
                else:
                    # Use Frechet only when there is enough motion history
                    if f < self.config.frechet_threshold:
                        adj[i, j] = adj[j, i] = 1
        return adj

    def find_connected_components(self, adj):
        n = adj.shape[0]
        visited = [False] * n
        groups = []

        def dfs(i, g):
            visited[i] = True
            g.append(i)
            for j in range(n):
                if adj[i, j] == 1 and not visited[j]:
                    dfs(j, g)

        for i in range(n):
            if not visited[i]:
                g = []
                dfs(i, g)
                groups.append(g)
        return groups

    def detect_groups(self, people: List[Person]) -> List[Group]:
        if len(people) == 0:
            return []

        adj = self.build_adjacency_matrix(people)
        member_groups = self.find_connected_components(adj)
        groups: List[Group] = []
        for members in member_groups:
            is_crowd = len(members) > self.config.crowd_threshold
            groups.append(Group(members=members, is_crowd=is_crowd))
        return groups


class Visualizer:
    def __init__(self, config: Config, traj_tracker: TrajectoryTracker):
        self.config = config
        self.traj_tracker = traj_tracker

    def get_group_color(self, group, group_idx, frame_count):
        if group.is_crowd:
            if group.alarm_triggered:
                cycle = frame_count % self.config.alarm_flash_rate
                return self.config.color_yellow if cycle < 5 else self.config.color_red
            else:
                return self.config.color_red
        elif len(group.members) == 1:
            return self.config.color_white
        else:
            return generate_group_color(group_idx)

    def draw_person(self, frame, person: Person, color, label: str):
        x1, y1, x2, y2 = person.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      color, self.config.box_thickness)

        fp = person.foot_point.astype(int)
        cv2.circle(frame, tuple(fp), 5, color, -1)

        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            self.config.text_scale, self.config.text_thickness
        )[0]

        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 10, y1),
            color, -1
        )
        cv2.putText(
            frame, label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, self.config.text_scale,
            self.config.color_black, self.config.text_thickness
        )

        traj = self.traj_tracker.get(person.id)
        if len(traj) > 1:
            pts = np.array([[int(p[0]), int(p[1])] for p in traj])
            cv2.polylines(frame, [pts], False,
                          color, self.config.trajectory_thickness)

    def draw_groups(self, frame, people, groups, frame_count):
        for idx, g in enumerate(groups):
            color = self.get_group_color(g, idx, frame_count)
            for p_idx in g.members:
                person = people[p_idx]
                if g.is_crowd:
                    if g.alarm_triggered:
                        status = f"ALARM! {g.crowd_duration:.1f}s"
                    else:
                        status = f"CROWD {g.crowd_duration:.1f}s"
                else:
                    status = f"G{idx + 1}" if len(g.members) > 1 else "Single"

                label = f"ID:{person.id} {status}"
                self.draw_person(frame, person, color, label)

    def draw_info_panel(self, frame, frame_count, num_people, num_crowds, num_alarms):
        info = [
            f"Frame: {frame_count}",
            f"People: {num_people}",
            f"Crowds: {num_crowds}",
            f"Active Alarms: {num_alarms}",
        ]
        y = 30
        for line in info:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        self.config.color_white, 2)
            y += 30

    def draw_alarm_warnings(self, frame, groups: List[Group]):
        h, w = frame.shape[:2]
        alarming = [g for g in groups if g.alarm_triggered]
        if not alarming:
            return

        cv2.putText(
            frame, f"!!! {len(alarming)} CROWD ALARM(S) !!!",
            (w // 2 - 250, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            self.config.color_red, 4
        )
        y_off = 100
        for i, g in enumerate(alarming, 1):
            text = f"Alarm {i}: {len(g.members)} people, {g.crowd_duration:.1f}s"
            cv2.putText(
                frame, text,
                (w // 2 - 200, y_off),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                self.config.color_orange, 2
            )
            y_off += 35


class CrowdGroupModel:
    """
    Streamlit-friendly crowd detection model.
    For each frame it returns ONE output image (same size as input)
    with:
      - group boxes + IDs + info panel
      - heatmap overlaid on top
    """
    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        model_path: str = "models/yolo11n.pt",   # change if you use another weights file
        config: Optional[Config] = None,
    ):
        self.config = config or Config()
        self.width = width
        self.height = height
        self.fps = fps if 0 < fps <= self.config.max_fps else self.config.default_fps

        # YOLO detection model
        self.model = YOLO(model_path)

        # DeepSort tracker
        self.tracker = DeepSort(
            max_age=self.config.tracker_max_age,
            n_init=self.config.tracker_n_init,
            max_iou_distance=self.config.tracker_max_iou_distance,
        )

        # helpers
        self.traj_tracker = TrajectoryTracker(self.config.trajectory_length)
        self.group_detector = GroupDetector(self.config, self.traj_tracker)
        self.group_timer_tracker = GroupTimerTracker(self.config.alert_trigger_seconds)
        self.visualizer = Visualizer(self.config, self.traj_tracker)
        self.heatmap_gen = HeatmapGenerator(width, height, self.config)

        self.frame_count = 0

    def detect_people(self, frame) -> List[Person]:
        """Run YOLO + DeepSort and return list of Person objects."""
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes:
                for box in r.boxes:
                    # cls 0 = person
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        w, h = x2 - x1, y2 - y1
                        conf = float(box.conf[0])
                        detections.append(([x1, y1, w, h], conf, 0))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        people: List[Person] = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, t.to_ltrb())
            bbox = (x1, y1, x2, y2)
            fp = calculate_foot_point(bbox)
            diag = calculate_diagonal_length(bbox)
            p = Person(
                id=t.track_id,
                bbox=bbox,
                foot_point=fp,
                diagonal_length=diag,
            )
            people.append(p)
            self.traj_tracker.update(p)

        return people

    def process_frame(self, frame):
        """
        Process a single frame:
          1. detect + track people
          2. find groups/crowds + update timers
          3. update heatmap
          4. draw boxes / labels / info panel
          5. overlay heatmap on annotated frame
        Returns: output_frame (same width/height as input)
        """
        # 1. detect & track
        people = self.detect_people(frame)

        # 2. group detection
        groups = self.group_detector.detect_groups(people)
        groups = self.group_timer_tracker.update(groups, people, self.fps)

        # 3. heatmap update
        self.heatmap_gen.update(people, groups)

        # 4. clean old trajectories
        active_ids = {p.id for p in people}
        self.traj_tracker.cleanup(active_ids)

        # 5. draw everything on annotated frame
        annotated = frame.copy()
        if groups:
            self.visualizer.draw_groups(annotated, people, groups, self.frame_count)

        num_crowds = sum(1 for g in groups if g.is_crowd)
        num_alarms = self.group_timer_tracker.get_active_alarm_count()
        has_alarm = self.group_timer_tracker.has_any_alarm()

        self.visualizer.draw_info_panel(
            annotated,
            self.frame_count,
            num_people=len(people),
            num_crowds=num_crowds,
            num_alarms=num_alarms,
        )

        if has_alarm:
            self.visualizer.draw_alarm_warnings(annotated, groups)

        # 6. overlay heatmap on top of annotated frame
        output_frame = self.heatmap_gen.generate_overlay(annotated)

        self.frame_count += 1
        return output_frame



