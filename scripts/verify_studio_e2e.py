"""
End-to-End QA Script for MangaVrse Reconstruction Studio:
Captures screenshots after each screen and each stage tab.
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ART_DIR = Path(r"C:\Users\mishr\.gemini\antigravity\brain\b62d04f8-4fe7-44ee-a0bc-09ae8092d0ca")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def run_studio_qa():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=d3d11"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # 1. Project List
        print("1. Capturing Project List...")
        page.goto("http://localhost:3000")
        page.wait_for_selector("text=Benchmark Test Suite")
        time.sleep(1.0)
        page.screenshot(path=str(ART_DIR / "studio_01_project_list.png"))

        # 2. Page Browser
        print("2. Opening Page Browser...")
        page.locator("text=Benchmark Test Suite").first.click()
        page.wait_for_selector("text=Page Browser")
        time.sleep(1.5)
        page.screenshot(path=str(ART_DIR / "studio_02_page_browser.png"))

        # 3. Enter Panel Workspace
        print("3. Opening Panel Workspace...")
        page.locator("text=Open →").first.click()
        page.wait_for_selector("text=1. Enhance")
        time.sleep(1.5)

        # Tab 1: Enhance
        print("4. Capturing Enhance Tab...")
        page.screenshot(path=str(ART_DIR / "studio_03_enhance_tab.png"))

        # Tab 2: Vectorize
        print("5. Capturing Vectorize Tab...")
        page.click("button:has-text('2. Vectorize')")
        time.sleep(1.5)
        page.screenshot(path=str(ART_DIR / "studio_04_vectorize_tab.png"))

        # Tab 3: Segment with Click-to-Segment Tool
        print("6. Capturing Segment Tab and testing Click-to-Segment Tool...")
        page.click("button:has-text('3. Segment')")
        time.sleep(1.5)

        img_loc = page.locator("img[alt='Segment Overlay']")
        box = img_loc.bounding_box()
        if box:
            # Positive click point
            page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
            time.sleep(1.0)
            # Negative click point
            page.click("button:has-text('- Background Point')")
            page.mouse.click(box["x"] + box["width"] * 0.2, box["y"] + box["height"] * 0.2)
            time.sleep(1.5)

        page.screenshot(path=str(ART_DIR / "studio_05_segment_refine.png"))

        # Tab 4: Layers
        print("7. Capturing Layers Tab...")
        page.click("button:has-text('4. Layers')")
        time.sleep(1.5)
        page.screenshot(path=str(ART_DIR / "studio_06_layers_tab.png"))

        # Tab 5: Colorize with Per-Layer Color Picker
        print("8. Capturing Colorize Tab and testing Color Picker...")
        page.click("button:has-text('5. Colorize')")
        time.sleep(1.5)

        color_input = page.locator("input[type='color']").first
        if color_input:
            color_input.fill("#00E5FF")
            time.sleep(1.0)

        page.screenshot(path=str(ART_DIR / "studio_07_colorize_override.png"))

        # Tab 6: Depth with Per-Layer Depth Slider
        print("9. Capturing Depth Tab and testing Depth Slider...")
        page.click("button:has-text('6. Depth')")
        time.sleep(1.5)

        depth_slider = page.locator("input[type='range'][step='0.01']").first
        if depth_slider:
            depth_slider.fill("0.85")
            time.sleep(1.0)

        page.screenshot(path=str(ART_DIR / "studio_08_depth_override.png"))

        # Tab 7: 2.5D Live Viewer
        print("10. Capturing Live 2.5D Viewer Tab...")
        page.click("button:has-text('7. 2.5D Viewer')")
        time.sleep(2.5)
        page.screenshot(path=str(ART_DIR / "studio_09_viewer_tab.png"))

        browser.close()
        print("Studio QA run successfully completed!")

if __name__ == "__main__":
    run_studio_qa()
