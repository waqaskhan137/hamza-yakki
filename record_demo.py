"""Record a demo video of form filling."""
import time
from playwright.sync_api import sync_playwright

SURVEY_URL = "https://survey.gslglobal.com/"

def record_demo():
    with sync_playwright() as playwright:
        # Launch browser with video recording
        browser = playwright.chromium.launch(headless=False, slow_mo=100)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            record_video_dir="demo_video",
            record_video_size={"width": 1280, "height": 900}
        )
        page = ctx.new_page()
        
        try:
            print("Recording demo video...")
            print("Navigating to survey...")
            page.goto(SURVEY_URL, wait_until="networkidle")
            time.sleep(3)
            
            # Accept terms
            print("Accepting terms...")
            try:
                page.get_by_text("I have read and accept", exact=False).first.click()
                time.sleep(1)
            except:
                pass
            
            # Click start
            for sel in [".sd-navigation__next-btn", ".sd-navigation__start-btn"]:
                try:
                    el = page.query_selector(sel)
                    if el and el.is_visible():
                        el.click()
                        time.sleep(2)
                        break
                except:
                    pass
            
            # Fill a few sample fields to show it working
            print("Filling sample fields...")
            
            # Try to fill email
            try:
                page.evaluate("""() => {
                    const q = window.survey.getQuestionByName('email');
                    if (q) window.survey.setValue('email', 'demo@example.com');
                }""")
                time.sleep(1)
            except:
                pass
            
            # Navigate a couple pages
            for i in range(3):
                print(f"Navigating to page {i+2}...")
                page.evaluate("() => window.survey.nextPage()")
                time.sleep(2)
            
            print("Demo complete! Closing browser...")
            time.sleep(2)
            
        finally:
            ctx.close()
            browser.close()
            
    print("Video saved to demo_video/ directory")

if __name__ == "__main__":
    record_demo()
