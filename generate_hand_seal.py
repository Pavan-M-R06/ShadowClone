"""
Generate a hand seal guide image (Naruto cross seal) using OpenCV drawing.
Creates: assets/hand_seal.png (with transparency)
"""

import numpy as np
import cv2
import os


def create_hand_seal_image(output_path, size=400):
    """
    Draw a stylized Naruto cross hand seal icon.
    Two crossed fingers forming the Shadow Clone seal.
    """
    # Create BGRA image (transparent background)
    img = np.zeros((size, size, 4), dtype=np.uint8)

    cx, cy = size // 2, size // 2
    scale = size / 400  # Base scale

    # ── Outer glow circle ──────────────────────────────────
    cv2.circle(img, (cx, cy), int(170 * scale), (255, 140, 0, 40), -1)
    cv2.circle(img, (cx, cy), int(140 * scale), (255, 165, 0, 60), -1)

    # ── Draw the "cross" hand seal shape ───────────────────

    # Colors
    skin_color = (130, 170, 220, 255)       # Skin tone (BGRA)
    outline_color = (0, 100, 200, 255)      # Dark orange outline
    finger_color = (140, 180, 230, 255)     # Lighter skin

    # ── LEFT HAND (horizontal bar of the cross) ───────────
    # Fist base (left side)
    fist_l_pts = np.array([
        [cx - int(120*scale), cy - int(25*scale)],
        [cx - int(120*scale), cy + int(25*scale)],
        [cx - int(30*scale),  cy + int(25*scale)],
        [cx - int(30*scale),  cy - int(25*scale)],
    ], np.int32)
    cv2.fillPoly(img, [fist_l_pts], skin_color)
    cv2.polylines(img, [fist_l_pts], True, outline_color, int(2*scale), cv2.LINE_AA)

    # Left hand index finger (extending right)
    finger_l1_pts = np.array([
        [cx - int(30*scale),  cy - int(22*scale)],
        [cx - int(30*scale),  cy - int(8*scale)],
        [cx + int(90*scale),  cy - int(8*scale)],
        [cx + int(95*scale),  cy - int(15*scale)],
        [cx + int(90*scale),  cy - int(22*scale)],
    ], np.int32)
    cv2.fillPoly(img, [finger_l1_pts], finger_color)
    cv2.polylines(img, [finger_l1_pts], True, outline_color, int(2*scale), cv2.LINE_AA)

    # Left hand middle finger (extending right, below index)
    finger_l2_pts = np.array([
        [cx - int(30*scale),  cy - int(5*scale)],
        [cx - int(30*scale),  cy + int(9*scale)],
        [cx + int(80*scale),  cy + int(9*scale)],
        [cx + int(85*scale),  cy + int(2*scale)],
        [cx + int(80*scale),  cy - int(5*scale)],
    ], np.int32)
    cv2.fillPoly(img, [finger_l2_pts], finger_color)
    cv2.polylines(img, [finger_l2_pts], True, outline_color, int(2*scale), cv2.LINE_AA)

    # ── RIGHT HAND (vertical bar of the cross) ────────────
    # Fist base (bottom)
    fist_r_pts = np.array([
        [cx - int(25*scale), cy + int(30*scale)],
        [cx + int(25*scale), cy + int(30*scale)],
        [cx + int(25*scale), cy + int(120*scale)],
        [cx - int(25*scale), cy + int(120*scale)],
    ], np.int32)
    cv2.fillPoly(img, [fist_r_pts], skin_color)
    cv2.polylines(img, [fist_r_pts], True, outline_color, int(2*scale), cv2.LINE_AA)

    # Right hand index finger (extending up)
    finger_r1_pts = np.array([
        [cx - int(22*scale), cy + int(30*scale)],
        [cx - int(22*scale), cy - int(90*scale)],
        [cx - int(15*scale), cy - int(95*scale)],
        [cx - int(8*scale),  cy - int(90*scale)],
        [cx - int(8*scale),  cy + int(30*scale)],
    ], np.int32)
    cv2.fillPoly(img, [finger_r1_pts], finger_color)
    cv2.polylines(img, [finger_r1_pts], True, outline_color, int(2*scale), cv2.LINE_AA)

    # Right hand middle finger (extending up, to the right of index)
    finger_r2_pts = np.array([
        [cx - int(5*scale),  cy + int(30*scale)],
        [cx - int(5*scale),  cy - int(80*scale)],
        [cx + int(2*scale),  cy - int(85*scale)],
        [cx + int(9*scale),  cy - int(80*scale)],
        [cx + int(9*scale),  cy + int(30*scale)],
    ], np.int32)
    cv2.fillPoly(img, [finger_r2_pts], finger_color)
    cv2.polylines(img, [finger_r2_pts], True, outline_color, int(2*scale), cv2.LINE_AA)

    # ── Center cross intersection highlight ────────────────
    cv2.rectangle(img,
                  (cx - int(25*scale), cy - int(22*scale)),
                  (cx + int(10*scale), cy + int(10*scale)),
                  (100, 140, 200, 200), -1)

    # ── Outer decorative ring ──────────────────────────────
    cv2.circle(img, (cx, cy), int(155*scale), (0, 140, 255, 180), int(3*scale), cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(160*scale), (0, 100, 200, 100), int(2*scale), cv2.LINE_AA)

    # ── Four small dots (cardinal directions - like a seal symbol) ──
    dot_r = int(6 * scale)
    dot_dist = int(145 * scale)
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        dx = int(cx + dot_dist * np.cos(rad))
        dy = int(cy + dot_dist * np.sin(rad))
        cv2.circle(img, (dx, dy), dot_r, (0, 160, 255, 220), -1, cv2.LINE_AA)

    # ── Text label ─────────────────────────────────────────
    label = "CLONE SEAL"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6 * scale
    thickness = max(1, int(2 * scale))
    text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    tx = (size - text_size[0]) // 2
    ty = size - int(20 * scale)

    # Shadow
    cv2.putText(img, label, (tx + 1, ty + 1), font, font_scale,
                (0, 0, 0, 200), thickness + 1, cv2.LINE_AA)
    # Text
    cv2.putText(img, label, (tx, ty), font, font_scale,
                (0, 180, 255, 255), thickness, cv2.LINE_AA)

    # ── Save ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"  ✓ Created: {output_path}")


if __name__ == '__main__':
    print("🖼️  Generating hand seal image...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output = os.path.join(base_dir, 'assets', 'hand_seal.png')
    create_hand_seal_image(output)
    print("\n✅ Hand seal image generated!")
