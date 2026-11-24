import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, channel='msedge')
        page = browser.new_page(viewport={'width': 1440, 'height': 900})

        # 1. Open home page
        print("Navigating to http://localhost:3000...")
        page.goto('http://localhost:3000', wait_until='networkidle')
        page.wait_for_timeout(1500)

        # 2. Click Benchmark Test Suite
        bench_card = page.locator("text='Benchmark Test Suite'").first
        if bench_card.is_visible():
            print("Clicking Benchmark Test Suite card...")
            bench_card.click()
            page.wait_for_timeout(2000)

        # 3. Inside PageBrowser, click the first detected panel
        first_panel = page.locator("aside div[class*='cursor-pointer']").first
        if first_panel.is_visible():
            print("Clicking first panel in PageBrowser sidebar...")
            first_panel.click()
            page.wait_for_timeout(2500)

        # 4. Check if in Panel Workspace and click 3. Segment Tab
        seg_button = page.locator("button:has-text('3. Segment')").first
        if seg_button.is_visible():
            print("Entering 3. Segment Tab...")
            seg_button.click()
            page.wait_for_timeout(2000)

            # Verify layer tree has character layers
            aside = page.locator("aside").first
            if aside.is_visible():
                print("Layer Tree Preview:\n", aside.inner_text()[:400])

            # Save segment screenshot
            artifact_shot = Path(r"C:\Users\mishr\.gemini\antigravity\brain\b62d04f8-4fe7-44ee-a0bc-09ae8092d0ca\studio_character_tracing_debug.png")
            page.screenshot(path=str(artifact_shot))
            print(f"Saved segment screenshot: {artifact_shot}")

        # 5. Enter 2.5D Viewer tab and test relative parallax
        viewer_tab = page.locator("button:has-text('7. 2.5D Viewer')").first
        if viewer_tab.is_visible():
            print("Switching to 7. 2.5D Viewer Tab...")
            viewer_tab.click()
            page.wait_for_timeout(3500)

            # Move mouse across canvas to trigger per-segment parallax
            canvas_elem = page.locator("canvas").first
            if canvas_elem.is_visible():
                cbox = canvas_elem.bounding_box()
                if cbox:
                    print("Testing relative parallax mouse motion...")
                    page.mouse.move(cbox['x'] + cbox['width'] * 0.2, cbox['y'] + cbox['height'] * 0.3)
                    page.wait_for_timeout(500)
                    page.mouse.move(cbox['x'] + cbox['width'] * 0.8, cbox['y'] + cbox['height'] * 0.7)
                    page.wait_for_timeout(1000)

            viewer_shot = Path(r"C:\Users\mishr\.gemini\antigravity\brain\b62d04f8-4fe7-44ee-a0bc-09ae8092d0ca\viewer_white_canvas_parallax.png")
            page.screenshot(path=str(viewer_shot))
            print(f"Saved viewer screenshot: {viewer_shot}")

        browser.close()

if __name__ == "__main__":
    run()
