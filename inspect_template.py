"""
inspect_template.py — Reads the uploaded PPTX template and prints
its slide count, layout names, placeholder names & positions.
"""
from pptx import Presentation
from pptx.util import Inches, Pt
import os

TEMPLATE = r"f:\AI Agents\trading-portal\PPT Template for Minor Project.pptx"

prs = Presentation(TEMPLATE)

print(f"Slide dimensions : {prs.slide_width.inches:.2f}\" x {prs.slide_height.inches:.2f}\"")
print(f"Total slides     : {len(prs.slides)}")
print(f"Slide layouts    : {len(prs.slide_layouts)}")
print()

print("=== SLIDE LAYOUTS ===")
for i, layout in enumerate(prs.slide_layouts):
    print(f"\n[{i}] Layout: '{layout.name}'")
    for ph in layout.placeholders:
        print(f"      PH idx={ph.placeholder_format.idx:2d}  type={ph.placeholder_format.type}  name='{ph.name}'")

print()
print("=== EXISTING SLIDES ===")
for si, slide in enumerate(prs.slides):
    print(f"\nSlide {si+1} — layout: '{slide.slide_layout.name}'")
    for ph in slide.placeholders:
        txt = ph.text_frame.text[:60].replace('\n',' ') if ph.has_text_frame else "—"
        print(f"  PH idx={ph.placeholder_format.idx:2d}  type={ph.placeholder_format.type}  '{ph.name}'  text='{txt}'")
