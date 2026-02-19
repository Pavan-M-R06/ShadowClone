"""
Shadow Clone AR — Main Application
===================================
Naruto-inspired AR app that detects the Shadow Clone hand seal
and spawns real-time mimicking clones behind the user.

Controls:
  - Make the cross hand seal → activates clones
  - Press 'D' → disable/dismiss clones
  - Press 'Q' → quit application

Requirements:
  pip install opencv-python mediapipe numpy
  python generate_sounds.py   (run once to create sound files)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import time
import winsound
import threading
import os
import sys


class ShadowCloneAR:
    """Main application class for Shadow Clone AR."""

    def __init__(self):
        # ── Paths ──────────────────────────────────────────────
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = os.path.join(self.base_dir, 'assets')

        # ── Camera ─────────────────────────────────────────────
        self.cap = None
        self.frame_w = 1280
        self.frame_h = 720

        # Try multiple camera indices and backends
        camera_options = [
            (0, cv2.CAP_DSHOW),    # Camera 0 with DirectShow (Windows)
            (0, cv2.CAP_ANY),      # Camera 0 with any backend
            (1, cv2.CAP_DSHOW),    # Camera 1 with DirectShow
            (1, cv2.CAP_ANY),      # Camera 1 with any backend
        ]

        for cam_id, backend in camera_options:
            print(f"   Trying camera {cam_id} (backend={backend})...")
            cap = cv2.VideoCapture(cam_id, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

                # Warm-up: read several frames to let the camera stabilize
                warmup_success = False
                for _ in range(10):
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        self.frame_h, self.frame_w = test_frame.shape[:2]
                        warmup_success = True
                        break
                    time.sleep(0.1)

                if warmup_success:
                    self.cap = cap
                    print(f"   ✅ Camera {cam_id} opened! ({self.frame_w}x{self.frame_h})")
                    break
                else:
                    cap.release()

        if self.cap is None:
            print("\n❌ ERROR: Could not open any camera!")
            print("   Please check:")
            print("   - Is your webcam connected?")
            print("   - Is another app using the camera? (Close Zoom, Teams, etc.)")
            print("   - Try running: python -c \"import cv2; c=cv2.VideoCapture(0); print(c.isOpened())\"")
            sys.exit(1)

        # ── MediaPipe Hand Detection ───────────────────────────
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.3,
        )

        # ── MediaPipe Selfie Segmentation ──────────────────────
        self.mp_seg = mp.solutions.selfie_segmentation
        self.selfie_seg = self.mp_seg.SelfieSegmentation(model_selection=1)

        # ── State Machine ──────────────────────────────────────
        self.clone_mode = False          # Are clones currently visible?
        self.jutsu_detected = False      # Was the seal just detected?
        self.jutsu_activation_time = 0   # When was the seal detected?
        self.show_text = False           # Show "KAGE BUNSHIN NO JUTSU!" ?
        self.text_start_time = 0
        self.poof_active = False         # Poof flash effect active?
        self.poof_start_time = 0
        self.poof_start_time = 0
        self.waiting_for_poof = False    # Waiting for delay before poof?

        # ── Rasengan State ─────────────────────────────────────
        self.rasengan_active = False     # Is the Rasengan visible?
        self.rasengan_charged = False    # Are hands in "charging" position?
        self.rasengan_sound_thread = None

        # ── Clone Configuration ────────────────────────────────
        self.num_clones = 9
        self.clone_offsets = self._generate_clone_positions()

        # ── Assets ─────────────────────────────────────────────
        self.hand_seal_img = self._load_hand_seal()
        self._verify_sounds()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SETUP HELPERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _generate_clone_positions(self):
        """
        Generate random scattered positions for clones across the background.
        Each clone appears at an independent random position, creating the
        illusion of separate clones standing around — not attached to the user.
        """
        import random
        random.seed(42)  # Consistent layout each run

        w, h = self.frame_w, self.frame_h
        offsets = []

        # Define zones where clones can appear (as % of frame)
        # Each clone gets a unique random position within its zone
        clone_zones = [
            # (x_range_min, x_range_max, y_range_min, y_range_max)
            # Far left clones
            (-0.60, -0.40, -0.20, 0.15),
            (-0.50, -0.30, -0.10, 0.20),
            (-0.45, -0.25,  0.00, 0.25),
            # Left-center clones
            (-0.30, -0.15, -0.25, 0.10),
            (-0.25, -0.10,  0.05, 0.20),
            # Right-center clones
            ( 0.10,  0.25, -0.25, 0.10),
            ( 0.15,  0.30,  0.05, 0.20),
            # Far right clones
            ( 0.25,  0.45, -0.20, 0.15),
            ( 0.30,  0.50, -0.10, 0.20),
            ( 0.40,  0.60,  0.00, 0.25),
            # Extra scattered clones (filling gaps)
            (-0.55, -0.35,  0.10, 0.30),
            ( 0.35,  0.55,  0.10, 0.30),
        ]

        for zone in clone_zones[:self.num_clones]:
            x_min, x_max, y_min, y_max = zone
            x_off = int(w * random.uniform(x_min, x_max))
            y_off = int(h * random.uniform(y_min, y_max))
            offsets.append((x_off, y_off))

        return offsets

    def _load_hand_seal(self):
        """Load the hand seal guide overlay (PNG with alpha)."""
        path = os.path.join(self.assets_dir, 'hand_seal.png')
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
                # Resize to a reasonable guide size
                target_h = max(1, int(self.frame_h * 0.10))
                aspect = img.shape[1] / img.shape[0]
                target_w = max(1, int(target_h * aspect))
                img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
                return img
        print("⚠  Hand seal image not found at:", path)
        return None

    def _verify_sounds(self):
        """Check that sound files exist."""
        for name in ['success.wav', 'poof.wav']:
            path = os.path.join(self.assets_dir, 'sounds', name)
            if not os.path.exists(path):
                print(f"⚠  Sound file missing: {path}")
                print("   Run: python generate_sounds.py")

    def play_rasengan_loop(self):
        """Play the Rasengan sound loop."""
        # Simple implementation: re-trigger every 2.8s
        # In a real app, we'd loop seamlessly.
        path = os.path.join(self.assets_dir, 'sounds', 'rasengan.wav')
        if not os.path.exists(path): return

        def _loop():
            while self.rasengan_active:
                try:
                    winsound.PlaySound(path, winsound.SND_FILENAME)
                except:
                    break
        
        if self.rasengan_sound_thread is None or not self.rasengan_sound_thread.is_alive():
            self.rasengan_sound_thread = threading.Thread(target=_loop, daemon=True)
            self.rasengan_sound_thread.start()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE B: HAND DETECTION — JUTSU ALGORITHM
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def draw_hand_landmarks(self, display, hand_results):
        """
        Draw green hand skeleton dots and connections on the display,
        matching the visual style from the reference video.
        """
        if not hand_results.multi_hand_landmarks:
            return

        mp_drawing = mp.solutions.drawing_utils
        mp_hands_style = mp.solutions.hands

        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                display,
                hand_landmarks,
                mp_hands_style.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 200, 0), thickness=2),
            )

    def is_jutsu_active(self, hand_results):
        """
        Detect the Naruto "Cross Seal" (Kage Bunshin hand sign).

        Uses the geometric cross-detection approach from the reference video:
          1. Two hands detected
          2. Both hands: index + middle fingers extended, ring + pinky curled
          3. Cross check: Index finger TIP of Hand 1 is near the Index finger
             BASE (MCP) of Hand 2, AND vice versa — confirming the fingers
             are actually interlocked in a cross shape.
        """
        if not hand_results.multi_hand_landmarks:
            return False
        if len(hand_results.multi_hand_landmarks) < 2:
            return False

        hand1 = hand_results.multi_hand_landmarks[0]
        hand2 = hand_results.multi_hand_landmarks[1]

        if not (self._check_fingers(hand1) and self._check_fingers(hand2)):
            return False

        # ── Cross Detection (from reference video) ─────────────
        # Check if index finger TIP of hand1 is near index finger BASE of hand2
        # Landmark 8  = Index Finger Tip
        # Landmark 5  = Index Finger MCP (base/knuckle)

        h1_tip = hand1.landmark[8]    # Hand 1 index tip
        h2_base = hand2.landmark[5]   # Hand 2 index base
        h2_tip = hand2.landmark[8]    # Hand 2 index tip
        h1_base = hand1.landmark[5]   # Hand 1 index base

        # Distance: Hand1 tip → Hand2 base
        dist_1to2 = ((h1_tip.x - h2_base.x) ** 2 +
                     (h1_tip.y - h2_base.y) ** 2) ** 0.5

        # Distance: Hand2 tip → Hand1 base
        dist_2to1 = ((h2_tip.x - h1_base.x) ** 2 +
                     (h2_tip.y - h1_base.y) ** 2) ** 0.5

        # ALSO check overall hand proximity (hands must be close together)
        h1_center_x = (h1_tip.x + h1_base.x) / 2
        h1_center_y = (h1_tip.y + h1_base.y) / 2
        h2_center_x = (h2_tip.x + h2_base.x) / 2
        h2_center_y = (h2_tip.y + h2_base.y) / 2

        hands_dist = ((h1_center_x - h2_center_x) ** 2 +
                      (h1_center_y - h2_center_y) ** 2) ** 0.5

        # The cross is detected if:
        # - At least one cross-distance is small (tips near opposing bases)
        # - AND the hands are close together overall
        cross_threshold = 0.15
        hands_threshold = 0.25

        cross_detected = (dist_1to2 < cross_threshold or
                          dist_2to1 < cross_threshold)
        hands_close = hands_dist < hands_threshold

        return cross_detected and hands_close

    def _check_fingers(self, hand):
        """
        Check that index + middle are extended,
        ring + pinky are curled.
        Uses landmark Y comparison (lower Y = higher on screen = extended).
        """
        lm = hand.landmark

        # Index finger: tip (8) above PIP joint (6) = extended
        index_extended = lm[8].y < lm[6].y

        # Middle finger: tip (12) above PIP joint (10) = extended
        middle_extended = lm[12].y < lm[10].y

        # Ring finger: tip (16) below PIP joint (14) = curled
        ring_curled = lm[16].y > lm[14].y

        # Pinky: tip (20) below PIP joint (18) = curled
        pinky_curled = lm[20].y > lm[18].y

        return index_extended and middle_extended and ring_curled and pinky_curled

    def _is_fist(self, hand):
        """Check if hand is a closed fist (all fingers curled)."""
        # Simple heuristic: Tip is CLOSE to MCP (knuckle)
        lm = hand.landmark
        threshold = 0.12  # Distance threshold
        
        # Check all 4 fingers (Index to Pinky)
        fingers_closed = True
        for i in [8, 12, 16, 20]:
            tip = lm[i]
            mcp = lm[i - 3] # Corresponding MCP
            dist = ((tip.x - mcp.x)**2 + (tip.y - mcp.y)**2)**0.5
            if dist > threshold:
                fingers_closed = False
                break
        return fingers_closed

    def _is_open_palm(self, hand):
        """Check if hand is open palm (all fingers extended)."""
        lm = hand.landmark
        # Check if tips are ABOVE PIP joints (for upright hand)
        # Better: Check if Tip-Wrist distance > PIP-Wrist distance
        wrist = lm[0]
        fingers_open = True
        for i in [8, 12, 16, 20]:
            tip = lm[i]
            pip = lm[i - 2]
            dist_tip = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
            dist_pip = ((pip.x - wrist.x)**2 + (pip.y - wrist.y)**2)**0.5
            if dist_tip < dist_pip * 1.2: # Tip should be significantly further
                fingers_open = False
                break
                fingers_open = False
                break
        return fingers_open

    def _is_namaste(self, left_hand, right_hand):
        """Check if hands are in 'Namaste' / Prayer position (palms together)."""
        # 1. Check if hands are close together
        l_wrist = left_hand.landmark[0]
        r_wrist = right_hand.landmark[0]
        dist = ((l_wrist.x - r_wrist.x)**2 + (l_wrist.y - r_wrist.y)**2)**0.5
        
        if dist > 0.2: return False # Hands too far apart

        # 2. Check if fingers are upright (tips above wrists)
        # In screen coord, y increases downwards. So Tip Y < Wrist Y
        if l_wrist.y < left_hand.landmark[8].y: return False
        if r_wrist.y < right_hand.landmark[8].y: return False
        
        return True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE C: SEGMENTATION & CLONE RENDERING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def segment_user(self, frame):
        """
        Use MediaPipe Selfie Segmentation to create a binary mask.
        Returns: mask (h, w) with 1 = user, 0 = background
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.selfie_seg.process(rgb)
        mask = (results.segmentation_mask > 0.6).astype(np.uint8)
        return mask

    def apply_anime_style(self, img):
        """
        Apply a 'Heavy Cartoon' / Anime style filter.
        1. Bilateral filter for smoothing colors (cartoon skin)
        2. Edge detection for outline (cel-shading)
        3. Quantize colors to reduce palette
        """
        # 1. Smoothing (Bilateral Filter) - keeps edges sharp
        # d=9, sigmaColor=75, sigmaSpace=75
        color = cv2.bilateralFilter(img, 9, 75, 75)

        # 2. Edge Detection (Adaptive Threshold)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blur, 255,
                                      cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 9)

        # 3. Combine Color + Edges
        # Create a 3-channel edge mask
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Cartoon effect: color image ANDed with edges
        cartoon = cv2.bitwise_and(color, edges_3ch)

        # Optional: Increase saturation for that "anime" pop
        hsv = cv2.cvtColor(cartoon, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, 1.2)  # Boost saturation
        s = np.clip(s, 0, 255).astype(np.uint8)
        hsv = cv2.merge([h, s, v])
        cartoon = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return cartoon



    def render_clones(self, frame, mask):
        """
        Render all shadow clones behind the real user.

        Process:
          1. Start with the original frame (background)
          2. Scale down the user silhouette to clone_scale (60%)
          3. For each clone offset, place the scaled clone
          4. Paint the real user on top at full opacity
        """
        h, w = frame.shape[:2]
        composite = frame.copy()

        # ── Pre-compute scaled-down clone image & mask ─────────
        clone_scale = 0.60  # Clones are 60% the size of real user
        small_w = int(w * clone_scale)
        small_h = int(h * clone_scale)

        # Scale down the user pixels and mask
        small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
        small_user = cv2.bitwise_and(small_frame, small_frame, mask=small_mask)

        # Apply Anime Style to the SCALED DOWN clone source
        small_user = self.apply_anime_style(small_user)

        # Center offset: place the scaled clone so its center aligns with offset
        cx_shift = (w - small_w) // 2
        cy_shift = (h - small_h) // 2

        # Sort clones by Y offset (back-to-front)
        sorted_offsets = sorted(self.clone_offsets, key=lambda o: o[1])

        for x_off, y_off in sorted_offsets:
            # Destination position for the scaled clone
            dst_x = cx_shift + x_off
            dst_y = cy_shift + y_off

            # Clip to frame boundaries
            # Source region (from small clone)
            sx1 = max(0, -dst_x)
            sy1 = max(0, -dst_y)
            sx2 = min(small_w, w - dst_x)
            sy2 = min(small_h, h - dst_y)

            # Destination region (on composite)
            dx1 = max(0, dst_x)
            dy1 = max(0, dst_y)
            dx2 = dx1 + (sx2 - sx1)
            dy2 = dy1 + (sy2 - sy1)

            region_w = sx2 - sx1
            region_h = sy2 - sy1

            if region_w <= 0 or region_h <= 0:
                continue

            # Extract regions
            clone_region = small_user[sy1:sy1 + region_h, sx1:sx1 + region_w]
            mask_region = small_mask[sy1:sy1 + region_h, sx1:sx1 + region_w]
            dst_region = composite[dy1:dy1 + region_h, dx1:dx1 + region_w]

            # Blend clone at 85% opacity
            clone_alpha = 0.85
            mask_3ch = np.stack([mask_region] * 3, axis=-1).astype(np.float32)

            blended = (
                dst_region.astype(np.float32) * (1 - clone_alpha * mask_3ch) +
                clone_region.astype(np.float32) * (clone_alpha * mask_3ch)
            ).astype(np.uint8)

            composite[dy1:dy1 + region_h, dx1:dx1 + region_w] = blended

        # Paint the REAL user on top at 100% opacity (full size)
        user_area = mask > 0
        composite[user_area] = frame[user_area]

        return composite

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PHASE D: UI OVERLAYS & EFFECTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def overlay_rgba(self, background, overlay, x, y, alpha_scale=1.0):
        """
        Overlay a BGRA image onto a BGR background at position (x, y).
        alpha_scale: additional opacity multiplier (0.0 – 1.0).
        """
        if overlay is None:
            return background

        oh, ow = overlay.shape[:2]
        bh, bw = background.shape[:2]

        # Calculate visible region
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(bw, x + ow), min(bh, y + oh)
        ox1, oy1 = x1 - x, y1 - y
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return background

        if overlay.shape[2] == 4:
            alpha = (overlay[oy1:oy2, ox1:ox2, 3] / 255.0) * alpha_scale
            alpha_3ch = np.stack([alpha] * 3, axis=-1)
            bg_region = background[y1:y2, x1:x2].astype(np.float32)
            fg_region = overlay[oy1:oy2, ox1:ox2, :3].astype(np.float32)
            background[y1:y2, x1:x2] = (
                alpha_3ch * fg_region + (1 - alpha_3ch) * bg_region
            ).astype(np.uint8)
        else:
            background[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]

        return background

    def draw_jutsu_text(self, display, elapsed):
        """
        Draw "KAGE BUNSHIN NO JUTSU!" with anime-style
        orange text + black outline, centered on screen.
        """
        text = "KAGE BUNSHIN NO JUTSU!"
        font = cv2.FONT_HERSHEY_DUPLEX
        h, w = display.shape[:2]

        # Pulsing scale effect
        base_scale = w / 640  # Scale text with resolution
        pulse = 1.0 + 0.05 * abs(np.sin(elapsed * 4))
        scale = base_scale * pulse
        thickness = max(2, int(base_scale * 2.5))

        text_size = cv2.getTextSize(text, font, scale, thickness)[0]
        tx = (w - text_size[0]) // 2
        ty = h // 3

        # Outer glow (large black outline)
        cv2.putText(display, text, (tx, ty), font, scale,
                    (0, 0, 0), thickness + 4, cv2.LINE_AA)
        # Inner glow (orange-yellow)
        cv2.putText(display, text, (tx, ty), font, scale,
                    (0, 140, 255), thickness, cv2.LINE_AA)

        # Secondary glow pass for "energy" feel
        glow_alpha = 0.3 * max(0, 1 - elapsed / 3.0)
        if glow_alpha > 0.01:
            glow = np.zeros_like(display)
            cv2.putText(glow, text, (tx, ty), font, scale * 1.02,
                        (0, 200, 255), thickness + 6, cv2.LINE_AA)
            glow = cv2.GaussianBlur(glow, (0, 0), 15)
            cv2.addWeighted(display, 1.0, glow, glow_alpha, 0, display)

    def draw_hand_seal_guide(self, display):
        """Draw the hand seal guide overlay in the top-right corner."""
        if self.hand_seal_img is None:
            return

        h, w = display.shape[:2]
        seal = self.hand_seal_img
        sh, sw = seal.shape[:2]

        # Position: top-right with 20px margin
        x = w - sw - 20
        y = 20

        # Semi-transparent border/backdrop
        pad = 10
        cv2.rectangle(display,
                      (x - pad, y - pad),
                      (x + sw + pad, y + sh + pad + 35),
                      (30, 30, 30), -1)
        cv2.rectangle(display,
                      (x - pad, y - pad),
                      (x + sw + pad, y + sh + pad + 35),
                      (0, 140, 255), 2)

        # Overlay the hand seal image
        self.overlay_rgba(display, seal, x, y, alpha_scale=0.85)

        # Instruction text below the seal
        label = "Make this seal!"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        lx = x + (sw - label_size[0]) // 2
        ly = y + sh + 25
        cv2.putText(display, label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)

    def draw_disable_button(self, display):
        """Draw the clone disable button indicator."""
        h, w = display.shape[:2]
        btn_w, btn_h = 250, 45
        bx = 20
        by = h - btn_h - 20

        # Button background
        overlay = display.copy()
        cv2.rectangle(overlay, (bx, by), (bx + btn_w, by + btn_h),
                      (0, 0, 160), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Button border
        cv2.rectangle(display, (bx, by), (bx + btn_w, by + btn_h),
                      (50, 50, 255), 2)

        # Button text
        cv2.putText(display, "Press 'D' to Release",
                    (bx + 12, by + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_poof_effect(self, display, elapsed):
        """
        Brief white flash that fades out — simulates smoke poof.
        Duration: ~0.5 seconds.
        """
        if elapsed < 0.5:
            intensity = 1.0 - (elapsed / 0.5)
            # Add white overlay with fading alpha
            white = np.full_like(display, 255)
            alpha = intensity * 0.65
            cv2.addWeighted(white, alpha, display, 1 - alpha, 0, display)

            # Draw expanding circles for "smoke" effect
            h, w = display.shape[:2]
            cx, cy = w // 2, h // 2
            num_puffs = 8
            for i in range(num_puffs):
                angle = (2 * np.pi * i) / num_puffs
                radius = int(50 + elapsed * 400)
                px = int(cx + np.cos(angle) * radius * 0.8)
                py = int(cy + np.sin(angle) * radius * 0.5)
                puff_r = int(40 + elapsed * 200)
                puff_alpha = max(0, intensity * 0.4)
                puff_overlay = display.copy()
                cv2.circle(puff_overlay, (px, py), puff_r,
                          (220, 220, 230), -1)
                cv2.addWeighted(puff_overlay, puff_alpha, display,
                               1 - puff_alpha, 0, display)

    def process_rasengan(self, display, hand_results):
        """
        Handle Rasengan logic:
        1. CHARGING: Both fists closed.
        2. ACTIVE: Right hand opens -> Rasengan appears.
        3. RESET: Right hand closes or disappears.
        4. STOP: Namaste (Prayer hands) -> Deactivate.
        """
        if not hand_results.multi_hand_landmarks:
            self.rasengan_active = False
            return

        # Identify hands
        left_hand = None
        right_hand = None

        for idx, hand_handedness in enumerate(hand_results.multi_handedness):
            label = hand_handedness.classification[0].label
            if label == 'Left': right_hand = hand_results.multi_hand_landmarks[idx]
            else: left_hand = hand_results.multi_hand_landmarks[idx]

        # 1. State Transitions
        if left_hand and right_hand:
            # Check for STOP (Namaste)
            if self._is_namaste(left_hand, right_hand):
                if self.rasengan_active:
                    self.rasengan_active = False # Deactivate!
                    self.rasengan_charged = True # Reset to charged state
                return

            # Check for Charging (Both Fists)
            if self._is_fist(left_hand) and self._is_fist(right_hand):
                self.rasengan_charged = True
        
        if self.rasengan_charged and right_hand:
            # Check for Activation (Right Open)
            if self._is_open_palm(right_hand):
                if not self.rasengan_active:
                    self.rasengan_active = True
                    self.play_rasengan_loop()
            elif self._is_fist(right_hand):
                 self.rasengan_active = False 

        # 2. Reset if Right hand lost?
        if not right_hand:
            self.rasengan_active = False
            if not left_hand:
                self.rasengan_charged = False

        # 3. Draw Rasengan if active
        if self.rasengan_active and right_hand:
            self.draw_rasengan(display, right_hand)

    def draw_rasengan(self, display, hand):
        """Draw spinning blue chakra ball on the palm."""
        lm = hand.landmark
        h, w = display.shape[:2]
        
        # Center = Midpoint between Wrist (0) and Middle MCP (9)
        wrist = lm[0]
        middle_mcp = lm[9]
        cx = int((wrist.x + middle_mcp.x) / 2 * w)
        cy = int((wrist.y + middle_mcp.y) / 2 * h)
        
        # Size based on hand size
        hand_size = ((wrist.x - middle_mcp.x)**2 + (wrist.y - middle_mcp.y)**2)**0.5 * w
        radius = int(hand_size * 1.3) # Smaller (was 2.5)
        
        # Draw Spinning Spirals
        t = time.time() * 20 # Fast spin
        
        overlay = display.copy()
        
        # Outer turbulence
        cv2.circle(overlay, (cx, cy), int(radius*1.2), (255, 200, 0), -1) # Faint outer cyan
        
        # Core layers
        cv2.circle(overlay, (cx, cy), radius, (255, 100, 0), -1) # Blue
        cv2.circle(overlay, (cx, cy), int(radius*0.7), (255, 255, 0), -1) # Cyan
        cv2.circle(overlay, (cx, cy), int(radius*0.4), (255, 255, 200), -1) # White core
        
        # Dynamic Spiral Lines
        for i in range(0, 360, 30):
            # Spirals expanding out
            angle_start = np.radians(i + t * 50)
            angle_end = np.radians(i + t * 50 + 120)
            
            p1x = int(cx + math.cos(angle_start) * radius * 0.2)
            p1y = int(cy + math.sin(angle_start) * radius * 0.2)
            
            p2x = int(cx + math.cos(angle_end) * radius * 1.0)
            p2y = int(cy + math.sin(angle_end) * radius * 1.0)
            
            cv2.line(overlay, (p1x, p1y), (p2x, p2y), (255, 255, 255), 2)

        # Alpha Blend
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # SOUND EFFECTS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def play_sound(self, sound_name):
        """Play a sound in a background thread (non-blocking)."""
        path = os.path.join(self.assets_dir, 'sounds', f'{sound_name}.wav')
        if not os.path.exists(path):
            return

        def _play():
            try:
                winsound.PlaySound(path, winsound.SND_FILENAME)
            except Exception:
                pass

        thread = threading.Thread(target=_play, daemon=True)
        thread.start()

    def draw_status_bar(self, display, fps):
        """Draw a minimal status bar at the bottom."""
        h, w = display.shape[:2]

        # FPS counter
        cv2.putText(display, f"FPS: {fps:.0f}",
                    (w - 120, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1, cv2.LINE_AA)

        # Clone count
        if self.clone_mode:
            cv2.putText(display, f"Clones: {self.num_clones}",
                        (w - 120, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    def run(self):
        """Main application loop."""
        window_name = 'Shadow Clone Jutsu AR'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        fps = 0
        prev_time = time.time()
        frame_count = 0

        print("\n🥷 Shadow Clone AR — Running!")
        print("   Make the cross hand seal to summon clones")
        print("   Press 'D' to dismiss clones")
        print("   Press 'Q' to quit\n")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Camera feed lost!")
                break

            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # ── FPS Calculation ────────────────────────────────
            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            # ── Phase B: Hand Detection (always active) ──────
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = self.hands.process(rgb)

            # Draw green hand skeleton (always visible)
            self.draw_hand_landmarks(display, hand_results)

            # Only check for jutsu trigger when clones are not active
            if not self.clone_mode and not self.waiting_for_poof:
                jutsu = self.is_jutsu_active(hand_results)

                if jutsu and not self.jutsu_detected:
                    # ── JUTSU TRIGGERED! ──
                    self.jutsu_detected = True
                    self.jutsu_activation_time = time.time()
                    self.show_text = True
                    self.text_start_time = time.time()
                    self.waiting_for_poof = True

                    # Play success sound
                    self.play_sound('success')

            # ── Waiting for poof delay ─────────────────────────
            if self.waiting_for_poof:
                elapsed_since_jutsu = time.time() - self.jutsu_activation_time
                if elapsed_since_jutsu >= 1.5:
                    # Activate clones!
                    self.clone_mode = True
                    self.waiting_for_poof = False
                    self.poof_active = True
                    self.poof_start_time = time.time()
                    self.play_sound('poof')

            # ── Phase C & D: Clone Rendering ───────────────────
            if self.clone_mode:
                mask = self.segment_user(frame)
                display = self.render_clones(frame, mask)



            # ── Process Rasengan (State & Visuals) ─────────────
            self.process_rasengan(display, hand_results)

            # ── Poof Effect ────────────────────────────────────
            if self.poof_active:
                poof_elapsed = time.time() - self.poof_start_time
                if poof_elapsed < 0.5:
                    self.draw_poof_effect(display, poof_elapsed)
                else:
                    self.poof_active = False

            # ── KAGE BUNSHIN text ──────────────────────────────
            if self.show_text:
                text_elapsed = time.time() - self.text_start_time
                if text_elapsed < 3.5:
                    self.draw_jutsu_text(display, text_elapsed)
                else:
                    self.show_text = False

            # ── Hand Seal Guide (only when clones inactive) ────
            if not self.clone_mode and not self.waiting_for_poof:
                self.draw_hand_seal_guide(display)

            # D key still works to disable clones (no visible button)

    # ── Status Bar ─────────────────────────────────────
            self.draw_status_bar(display, fps)

            # ── Display ────────────────────────────────────────
            cv2.imshow(window_name, display)

            # ── Key Handling ───────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('d') or key == ord('D'):
                if self.clone_mode:
                    self.clone_mode = False
                    self.jutsu_detected = False
                    self.show_text = False
                    print("   💨 Clones dismissed!")

        # ── Cleanup ────────────────────────────────────────────
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        self.selfie_seg.close()
        print("\n🥷 Shadow Clone AR — Closed. See ya!")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ENTRY POINT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == '__main__':
    print("=" * 50)
    print("  🥷 SHADOW CLONE AR — Naruto Jutsu Detector")
    print("=" * 50)

    app = ShadowCloneAR()
    app.run()
