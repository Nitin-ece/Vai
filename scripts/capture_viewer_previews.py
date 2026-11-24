"""
Script to capture screenshots and QA the MangaVrse Three.js Viewer across all 5 benchmark scenes.
"""
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

ART_DIR = Path(r"C:\Users\mishr\.gemini\antigravity\brain\b62d04f8-4fe7-44ee-a0bc-09ae8092d0ca")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def run_qa():
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-gl=angle", "--use-angle=d3d11"],
        )
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("Navigating to http://localhost:3000...")
        page.goto("http://localhost:3000")
        page.wait_for_selector("canvas", timeout=15000)
        time.sleep(2.5)  # Let textures load and Three.js render

        # 1. Clean Lineart (Initial)
        print("Capturing clean_lineart...")
        page.screenshot(path=str(ART_DIR / "viewer_clean_lineart.png"))

        # 2. Test GSAP Push-In
        print("Testing GSAP Camera Push-In...")
        page.click("button:has-text('Push-In')")
        time.sleep(3.5)  # Wait for push-in transition
        page.screenshot(path=str(ART_DIR / "viewer_push_in.png"))

        # Reset camera
        page.click("button:has-text('Reset')")
        time.sleep(1.5)

        # 3. Dense Screentone
        print("Switching to Dense Screentone...")
        page.click("button:has-text('Dense Screentone')")
        time.sleep(2.0)
        page.screenshot(path=str(ART_DIR / "viewer_dense_screentone.png"))

        # 4. Low-Res Scan
        print("Switching to Low-Res Scan...")
        page.click("button:has-text('Low-Res Scan')")
        time.sleep(2.0)
        page.screenshot(path=str(ART_DIR / "viewer_lowres_scan.png"))

        # 5. Speed Lines
        print("Switching to Speed Lines...")
        page.click("button:has-text('Speed Lines')")
        time.sleep(2.0)
        page.screenshot(path=str(ART_DIR / "viewer_speed_lines.png"))

        # 6. Text Heavy
        print("Switching to Text Heavy...")
        page.click("button:has-text('Text Heavy')")
        time.sleep(2.0)
        page.screenshot(path=str(ART_DIR / "viewer_text_heavy.png"))

        # 7. Mouse Parallax Motion Test
        print("Testing mouse-driven parallax motion...")
        canvas = page.locator("canvas")
        box = canvas.bounding_box()
        if box:
            cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            for offset_x in [-200, -100, 0, 100, 200, 100, 0, -100]:
                for offset_y in [-150, 0, 150]:
                    page.mouse.move(cx + offset_x, cy + offset_y)
                    time.sleep(0.04)

        # Capture parallax angle view
        page.screenshot(path=str(ART_DIR / "viewer_parallax_angle.png"))

        # Check FPS text from indicator
        fps_text = page.locator("header span:has-text('FPS')").inner_text()
        print(f"Reported performance: {fps_text}")

        browser.close()
        print("All screenshots successfully captured!")

if __name__ == "__main__":
    run_qa()
